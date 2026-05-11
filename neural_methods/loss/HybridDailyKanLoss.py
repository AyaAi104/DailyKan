"""Hybrid anti-collapse loss for DailyKan."""

import torch
from torch import nn


class HybridDailyKanLoss(nn.Module):
    """Pearson + variance + in-band SNR + spectral KL loss for rPPG.

    Args:
        fs: Sampling rate in Hz.
        low_hz: Lower physiological band bound.
        high_hz: Upper physiological band bound.
        w_var: Variance-matching weight.
        w_snr: In-band SNR weight.
        w_freq: Frequency distribution KL weight.

    Inputs:
        pred: Predicted waveform, shape [B, T], already z-score normalized.
        label: Target waveform, shape [B, T], already z-score normalized.

    Returns:
        total_loss, component_dict
    """

    def __init__(self, fs=30, low_hz=0.7, high_hz=3.0,
                 w_var=0.3, w_snr=0.1, w_freq=0.2):
        super().__init__()
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.w_var = w_var
        self.w_snr = w_snr
        self.w_freq = w_freq
        self.eps = 1e-6
        self.kl_eps = 1e-8

    def _pearson_loss(self, pred, label):
        pred_centered = pred - torch.mean(pred, dim=-1, keepdim=True)
        label_centered = label - torch.mean(label, dim=-1, keepdim=True)
        numerator = torch.sum(pred_centered * label_centered, dim=-1)
        pred_energy = torch.sum(pred_centered.pow(2), dim=-1)
        label_energy = torch.sum(label_centered.pow(2), dim=-1)
        denominator = torch.sqrt(
            pred_energy.clamp_min(self.eps) * label_energy.clamp_min(self.eps)
        )
        corr = numerator / denominator
        return torch.mean(1.0 - corr)

    def _psd(self, x):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        window = torch.hann_window(
            x.shape[-1],
            periodic=True,
            dtype=x.dtype,
            device=x.device,
        )
        spectrum = torch.fft.rfft(x * window, dim=-1)
        return spectrum.abs().pow(2)

    def _band_mask(self, length, device):
        freqs = torch.fft.rfftfreq(length, d=1.0 / float(self.fs)).to(device)
        return (freqs >= self.low_hz) & (freqs <= self.high_hz)

    def _normalized_band_psd(self, x):
        psd = self._psd(x)
        mask = self._band_mask(x.shape[-1], x.device)
        psd = psd * mask.to(dtype=psd.dtype).unsqueeze(0)
        return psd / psd.sum(dim=-1, keepdim=True).clamp_min(self.kl_eps)

    def _snr_loss(self, pred):
        psd = self._psd(pred)
        mask = self._band_mask(pred.shape[-1], pred.device)
        in_band = psd[:, mask].sum(dim=-1).clamp_min(self.eps)
        out_band = psd[:, ~mask].sum(dim=-1).clamp_min(self.eps)
        return torch.mean(-torch.log(in_band / out_band + self.eps))

    def _freq_kl_loss(self, pred, label):
        pred_psd = self._normalized_band_psd(pred)
        label_psd = self._normalized_band_psd(label)
        kl_pred_label = pred_psd * (
            torch.log(pred_psd + self.kl_eps) - torch.log(label_psd + self.kl_eps)
        )
        kl_label_pred = label_psd * (
            torch.log(label_psd + self.kl_eps) - torch.log(pred_psd + self.kl_eps)
        )
        return torch.mean(0.5 * (
            kl_pred_label.sum(dim=-1) + kl_label_pred.sum(dim=-1)
        ))

    def forward(self, pred, label):
        pearson = self._pearson_loss(pred, label)
        pred_std = torch.std(pred, dim=-1, unbiased=False).clamp_min(self.eps)
        label_std = torch.std(label, dim=-1, unbiased=False).clamp_min(self.eps)
        var = torch.mean(torch.abs(torch.log(pred_std) - torch.log(label_std)))
        snr = self._snr_loss(pred)
        freq = self._freq_kl_loss(pred, label)
        total = pearson + self.w_var * var + self.w_snr * snr + self.w_freq * freq

        parts = {
            "pearson": pearson.detach(),
            "var": var.detach(),
            "snr": snr.detach(),
            "freq": freq.detach(),
            "total": total.detach(),
        }
        return total, parts


if __name__ == "__main__":
    torch.manual_seed(7)
    loss_fn = HybridDailyKanLoss(fs=30)

    constant_pred = torch.zeros(2, 180)
    random_label = torch.randn(2, 180)
    _, const_parts = loss_fn(constant_pred, random_label)
    print("constant var:", float(const_parts["var"]))
    assert float(const_parts["var"]) > 5.0

    t = torch.arange(180, dtype=torch.float32) / 30.0
    matched = torch.sin(2.0 * torch.pi * 1.0 * t) + torch.sin(2.0 * torch.pi * 5.0 * t)
    matched = matched.unsqueeze(0).repeat(2, 1)
    total, matched_parts = loss_fn(matched, matched)
    print("matched total:", float(total))
    print({k: float(v) for k, v in matched_parts.items()})
    assert abs(float(total)) < 0.2
