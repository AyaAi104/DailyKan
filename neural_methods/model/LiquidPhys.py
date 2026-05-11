"""LiquidPhys: Liquid neural rPPG with pose and optical-flow guidance.

LiquidPhys consumes a face-video clip plus compact motion descriptors:

    frames: [B, C, T, H, W] or [B, T, C, H, W]
    pose:   [B, 3, T] or [B, T, 3] yaw, pitch, roll
    flow:   [B, K, T] or [B, T, K] compact optical-flow features

The model first encodes RGB/video channels into one embedding per frame. The
temporal core is an ncps CfC layer. The ncps public CfC API exposes the input
sequence and hidden states, but not the internal time constant parameters
directly. LiquidPhys therefore uses two motion hooks:

1. a motion embedding concatenated to each CfC input step, allowing CfC gates to
   condition their continuous-time dynamics on pose and flow;
2. a learned hidden-state update gate that blends each CfC output with the
   previous gated state. Small gate values behave like a larger effective time
   constant: the state changes more slowly during large or difficult motion.

This keeps pose and flow auxiliary. They can modulate temporal integration, but
the BVP estimate is still driven by the video encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ncps.torch import CfC
except ImportError:  # pragma: no cover - exercised only when dependency missing.
    CfC = None


class LiquidPhys(nn.Module):
    """Pose and optical-flow guided liquid rPPG model.

    Args:
        in_channels: Number of video channels after preprocessing.
        embed_dim: Per-frame video embedding size.
        hidden_dim: CfC hidden size.
        motion_dim: pose_dim + flow_dim.
        motion_embed_dim: Auxiliary motion embedding concatenated to video.
        dropout: Dropout probability before the BVP head.
        cfc_mode: ncps CfC mode: "default", "pure", or "no_gate".
        mixed_memory: Whether to use the CfC mixed-memory variant.

    Returns:
        rPPG waveform with shape [B, T].
    """

    def __init__(
        self,
        in_channels=6,
        embed_dim=64,
        hidden_dim=64,
        motion_dim=10,
        motion_embed_dim=16,
        dropout=0.1,
        cfc_mode="default",
        mixed_memory=False,
    ):
        super().__init__()
        if CfC is None:
            raise ImportError(
                "LiquidPhys requires ncps. Install it with: pip install ncps"
            )
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.motion_dim = motion_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.SiLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.motion_norm = nn.LayerNorm(motion_dim)
        self.motion_mlp = nn.Sequential(
            nn.Linear(motion_dim, 32),
            nn.SiLU(),
            nn.Linear(32, motion_embed_dim),
            nn.SiLU(),
        )
        self.liquid = CfC(
            input_size=embed_dim + motion_embed_dim,
            units=hidden_dim,
            return_sequences=True,
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=cfc_mode,
            backbone_units=128,
            backbone_layers=1,
            backbone_dropout=dropout,
        )
        self.update_gate = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.bvp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _frames_to_bcthw(self, frames):
        if frames.ndim != 5:
            raise ValueError(f"LiquidPhys expects 5-D frames, got {frames.shape}")
        if frames.shape[1] == self.in_channels:
            return frames
        if frames.shape[2] == self.in_channels:
            return frames.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(
            f"Cannot infer video channel axis for frames {frames.shape}; "
            f"expected {self.in_channels} channels."
        )

    @staticmethod
    def _aux_to_btf(aux, feature_dim, name, target_t):
        if aux.ndim != 3:
            raise ValueError(f"LiquidPhys expects {name} as 3-D tensor, got {aux.shape}")
        if aux.shape[-1] == feature_dim:
            aux = aux
        elif aux.shape[1] == feature_dim:
            aux = aux.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Cannot infer feature axis for {name} {aux.shape}")
        if aux.shape[1] != target_t:
            aux = F.interpolate(
                aux.permute(0, 2, 1),
                size=target_t,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1).contiguous()
        return aux

    @staticmethod
    def _apply_motion_update_gate(liquid_seq, update_gate):
        gated_states = [liquid_seq[:, 0]]
        for t in range(1, liquid_seq.shape[1]):
            gate_t = update_gate[:, t]
            gated_states.append(gate_t * liquid_seq[:, t] + (1.0 - gate_t) * gated_states[-1])
        return torch.stack(gated_states, dim=1)

    def forward(self, frames, pose, flow):
        frames = self._frames_to_bcthw(frames)
        batch_size, _, length, _, _ = frames.shape

        pose = self._aux_to_btf(pose, 3, "pose", length)
        flow_dim = self.motion_dim - 3
        flow = self._aux_to_btf(flow, flow_dim, "flow", length)
        motion = self.motion_norm(torch.cat([pose, flow], dim=-1))

        encoded = self.encoder(frames)
        if encoded.shape[2] != length:
            encoded = F.interpolate(
                encoded,
                size=(length, encoded.shape[3], encoded.shape[4]),
                mode="trilinear",
                align_corners=False,
            )
        encoded = F.adaptive_avg_pool3d(encoded, (length, 1, 1)).view(
            batch_size, self.embed_dim, length
        )
        encoded = encoded.transpose(1, 2).contiguous()

        motion_embed = self.motion_mlp(motion)
        liquid_input = torch.cat([encoded, motion_embed], dim=-1)
        liquid_seq, _ = self.liquid(liquid_input)
        update_gate = self.update_gate(motion)
        gated_seq = self._apply_motion_update_gate(liquid_seq, update_gate)
        rppg = self.bvp_head(self.dropout(gated_seq)).squeeze(-1)
        return rppg

    def diagnose(self, frames, pose, flow):
        """Return tensor shapes from a no-grad diagnostic forward pass."""
        with torch.no_grad():
            frames = self._frames_to_bcthw(frames)
            length = frames.shape[2]
            pose = self._aux_to_btf(pose, 3, "pose", length)
            flow = self._aux_to_btf(flow, self.motion_dim - 3, "flow", length)
            pred = self.forward(frames, pose, flow)
        return {
            "frames": tuple(frames.shape),
            "pose": tuple(pose.shape),
            "flow": tuple(flow.shape),
            "pred": tuple(pred.shape),
        }
