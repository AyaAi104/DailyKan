"""DailyKan: pose-guided KAN model for motion-robust rPPG.

The model consumes:
    frames: [B, C, T, H, W]
    pose:   [B, 3, T] in yaw, pitch, roll order

It uses lightweight KAN channel mixers and pose-conditioned modulation to
suppress motion-correlated components before predicting one rPPG value per
frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """Radial-basis KAN linear layer.

    This is a compact, dependency-free KAN variant. Each scalar input dimension
    is expanded over a fixed grid, giving learnable univariate functions whose
    outputs are mixed into the target features.
    """

    def __init__(self, in_features, out_features, grid_size=8, grid_range=(-2.0, 2.0),
                 base_activation=nn.SiLU, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("grid", grid)
        self.inv_denom = 1.0 / ((grid[1] - grid[0]).item() + 1e-6)

        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )
        self.activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base.weight, a=5 ** 0.5)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def forward(self, x):
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        base_out = self.base(self.activation(x_flat))

        basis = torch.exp(-((x_flat.unsqueeze(-1) - self.grid) * self.inv_denom) ** 2)
        spline_out = torch.einsum("big,oig->bo", basis, self.spline_weight)
        out = base_out + spline_out
        return out.reshape(*original_shape, self.out_features)


class KANPointwise3D(nn.Module):
    """Apply a KAN channel projection at each spatiotemporal location."""

    def __init__(self, in_channels, out_channels, grid_size=8):
        super().__init__()
        self.kan = KANLinear(in_channels, out_channels, grid_size=grid_size)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.kan(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class TemporalDifferenceConv3D(nn.Module):
    """Temporal central-difference 3D convolution for motion-robust features.

    The effective operator is:
        standard_conv(x) - sigmoid(theta) * central_difference_term(x)

    This is equivalent to (1 - theta) * standard_conv + theta * CDC, where
    CDC = standard_conv - central_difference_term.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, theta=0.5):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("TemporalDifferenceConv3D expects an odd kernel size.")
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        theta = min(max(theta, 1e-4), 1.0 - 1e-4)
        self.theta_logit = nn.Parameter(torch.logit(torch.tensor(theta)))
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        standard_out = self.conv(x)
        theta = torch.sigmoid(self.theta_logit)
        kernel_diff = self.conv.weight.sum(dim=(2, 3, 4), keepdim=True)
        diff_out = F.conv3d(
            x,
            kernel_diff,
            bias=None,
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        x = standard_out - theta * diff_out
        return self.act(self.norm(x))


class KANVideoBlock(nn.Module):
    """Spatial downsample plus KAN channel mixing."""

    def __init__(self, in_channels, out_channels, grid_size=8, spatial_stride=2):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(1, spatial_stride, spatial_stride),
                                 stride=(1, spatial_stride, spatial_stride))
        self.mix = KANPointwise3D(in_channels, out_channels, grid_size=grid_size)
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.mix(x)
        return self.act(self.norm(x))


class PoseKANEncoder(nn.Module):
    """Encode pose trajectory and produce feature-wise modulation."""

    def __init__(self, channels, hidden=32, grid_size=8):
        super().__init__()
        self.pose_norm = nn.BatchNorm1d(3)
        self.pose_proj = nn.Sequential(
            KANLinear(3, hidden, grid_size=grid_size),
            nn.SiLU(),
            KANLinear(hidden, channels * 3, grid_size=grid_size),
        )
        self.channels = channels

    def forward(self, pose):
        # pose: [B, 3, T] -> [B, T, 3]
        pose = self.pose_norm(pose).transpose(1, 2)
        params = self.pose_proj(pose)
        gamma, beta, gate = torch.chunk(params, 3, dim=-1)
        gamma = torch.tanh(gamma).transpose(1, 2)
        beta = beta.transpose(1, 2)
        gate = torch.sigmoid(gate).transpose(1, 2)
        return gamma, beta, gate


class PoseGuidedKANFusion(nn.Module):
    """Modulate ROI features using pose-derived KAN gates."""

    def __init__(self, channels, pose_hidden=32, grid_size=8):
        super().__init__()
        self.pose_encoder = PoseKANEncoder(channels, pose_hidden, grid_size)
        self.refine = nn.Sequential(
            KANLinear(channels, channels, grid_size=grid_size),
            nn.SiLU(),
            KANLinear(channels, channels, grid_size=grid_size),
        )

    def forward(self, roi_feat, pose):
        # roi_feat: [B, C, T]
        gamma, beta, gate = self.pose_encoder(pose)
        modulated = roi_feat * (1.0 + 0.5 * gamma) + 0.1 * beta
        suppressed = modulated * gate
        refined = self.refine(suppressed.transpose(1, 2)).transpose(1, 2)
        return refined + roi_feat


class PoseSpatialAttention(nn.Module):
    """Pose-guided channel-time attention before spatial pooling."""

    def __init__(self, channels, pose_hidden=32, grid_size=8):
        super().__init__()
        self.pose_norm = nn.BatchNorm1d(3)
        self.pose_gate = nn.Sequential(
            KANLinear(3, pose_hidden, grid_size=grid_size),
            nn.SiLU(),
            KANLinear(pose_hidden, channels, grid_size=grid_size),
        )

    def forward(self, x, pose):
        if pose.shape[-1] != x.shape[2]:
            pose = F.interpolate(pose, size=x.shape[2], mode="linear", align_corners=False)
        pose = self.pose_norm(pose).transpose(1, 2)
        gate = torch.sigmoid(self.pose_gate(pose)).transpose(1, 2)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        return x + 0.5 * (gate * x - x)


class KANTemporalHead(nn.Module):
    """Temporal smoothing and per-frame rPPG regression."""

    def __init__(self, channels, grid_size=8):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            KANLinear(channels, channels // 2, grid_size=grid_size),
            nn.SiLU(),
            KANLinear(channels // 2, 1, grid_size=grid_size),
        )

    def forward(self, x):
        x = self.temporal(x) + x
        y = self.head(x.transpose(1, 2)).squeeze(-1)
        return y


class DailyKan(nn.Module):
    """DailyKan rPPG network."""

    def __init__(self, in_channels=6, frames=180, width=32, grid_size=8,
                 pose_hidden=32, dropout=0.1):
        super().__init__()
        self.frames = frames
        self.stem = nn.Sequential(
            TemporalDifferenceConv3D(in_channels, in_channels, kernel_size=3, theta=0.5),
            KANVideoBlock(in_channels, width, grid_size=grid_size, spatial_stride=2),
            KANVideoBlock(width, width * 2, grid_size=grid_size, spatial_stride=2),
            KANVideoBlock(width * 2, width * 2, grid_size=grid_size, spatial_stride=2),
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.pose_spatial_attention = PoseSpatialAttention(width * 2, pose_hidden, grid_size)
        self.fusion = PoseGuidedKANFusion(width * 2, pose_hidden, grid_size)
        self.dropout = nn.Dropout(dropout)
        self.head = KANTemporalHead(width * 2, grid_size)

    def forward(self, frames, pose_angles):
        if frames.ndim != 5:
            raise ValueError(f"DailyKan expects frames [B,C,T,H,W], got {frames.shape}")
        if pose_angles.ndim != 3:
            raise ValueError(f"DailyKan expects pose [B,3,T], got {pose_angles.shape}")

        b, _, t, _, _ = frames.shape
        if pose_angles.shape[-1] != t:
            pose_angles = F.interpolate(pose_angles, size=t, mode="linear", align_corners=False)

        x = self.stem(frames)
        x = self.pose_spatial_attention(x, pose_angles)
        if x.shape[2] != t:
            x = F.interpolate(x, size=(t, x.shape[3], x.shape[4]), mode="trilinear", align_corners=False)
        x = F.adaptive_avg_pool3d(x, (t, 1, 1)).view(b, -1, t)
        x = self.fusion(x, pose_angles)
        x = self.dropout(x)
        rppg = self.head(x)
        return rppg
