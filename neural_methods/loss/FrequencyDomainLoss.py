"""Frequency-domain loss for rPPG waveform training."""

import torch
from torch import nn


class FrequencyDomainLoss(nn.Module):
    """Symmetric KL loss between band-limited rPPG power spectra.

    Args:
        fs: Sampling rate in Hz.
        low_hz: Lower frequency bound. Default 0.7 Hz, 42 BPM.
        high_hz: Upper frequency bound. Default 4.0 Hz, 240 BPM.

    Inputs:
        pred: Predicted waveform, shape [B, T].
        label: Target waveform, shape [B, T].
    """

    def __init__(self, fs=30, low_hz=0.7, high_hz=4.0):
        super().__init__()
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.eps = 1e-8

    def _band_limited_psd(self, x):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        window = torch.hann_window(
            x.shape[-1],
            periodic=True,
            dtype=x.dtype,
            device=x.device,
        )
        spectrum = torch.fft.rfft(x * window, dim=-1)
        psd = spectrum.abs().pow(2)

        freqs = torch.fft.rfftfreq(
            x.shape[-1],
            d=1.0 / float(self.fs),
        ).to(x.device)
        mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        psd = psd * mask.to(dtype=psd.dtype).unsqueeze(0)
        psd = psd / psd.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return psd

    def forward(self, pred, label):
        pred_psd = self._band_limited_psd(pred)
        label_psd = self._band_limited_psd(label)

        kl_pred_label = pred_psd * (
            torch.log(pred_psd + self.eps) - torch.log(label_psd + self.eps)
        )
        kl_label_pred = label_psd * (
            torch.log(label_psd + self.eps) - torch.log(pred_psd + self.eps)
        )
        loss = 0.5 * (
            kl_pred_label.sum(dim=-1) + kl_label_pred.sum(dim=-1)
        )
        return loss.mean()


if __name__ == "__main__":
    pred = torch.randn(2, 180)
    label = torch.randn(2, 180)
    criterion = FrequencyDomainLoss(fs=30)
    print(criterion(pred, label))
