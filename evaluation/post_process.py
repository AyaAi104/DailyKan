"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, power2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, cheby2, sosfiltfilt, welch as scipy_welch
from scipy.sparse import spdiags
from copy import deepcopy

# 0 = Butterworth (original toolbox)
# 1 = Chebyshev II
# 2 = Welch PSD method (from SQI script: butter(4) wide bandpass + Welch + argmax)
evaluate_butterworth: int = 0

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)

# modify $1, find final peek here:
def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.80, high_pass=1.667):
    # Note: to more closely match results in the NeurIPS 2023 toolbox paper,
    # we recommend low_pass=0.75 and high_pass=2.5 instead of the defaults above.
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)

    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_welch_hr(ppg_signal, fs=50, low_pass=0.80, high_pass=1.667, nperseg=1024):
    """Calculate heart rate using Welch PSD method (from SQI script).

    Uses Welch's method for smoother PSD estimation, more robust against noise
    than single-window periodogram.
    """
    ppg_signal = np.asarray(ppg_signal, dtype=float).flatten()
    x = ppg_signal - np.mean(ppg_signal)  # zero-mean

    nperseg_actual = min(nperseg, len(x))
    if nperseg_actual < 8:
        return 0.0

    f, Pxx = scipy_welch(x, fs=fs, nperseg=nperseg_actual)

    hr_mask = (f >= low_pass) & (f <= high_pass)
    if not np.any(hr_mask):
        return 0.0

    peak_freq = f[hr_mask][np.argmax(Pxx[hr_mask])]
    return peak_freq * 60.0


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_ppg_signal(np.array): predicted PPG signal
            label_ppg_signal(np.array): ground truth, label PPG signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc
#$3
def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.6, high_pass=3.3):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.6 Hz
        to 3.3 Hz.

        Ref for low_pass and high_pass filters:
        R. Cassani, A. Tiwari and T. H. Falk, "Optimal filter characterization for photoplethysmography-based pulse rate and
        pulse power spectrum estimation," 2020 IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada,
        doi: 10.1109/EMBC44109.2020.9175396.

        Note: to more closely match results in the NeurIPS 2023 toolbox paper, we recommend low_pass=0.75 and high_pass=2.5
        instead of the defaults above.

        Args:
            pred_ppg_signal(np.array): predicted PPG signal
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        if evaluate_butterworth == 0:
            # Butterworth 4th order, narrow band
            [b, a] = butter(4, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
            predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
            labels = scipy.signal.filtfilt(b, a, np.double(labels))
            print("\nrppg-toolbox Filter: Butterworth(4), fl=0.8, fh=3.3")

        elif evaluate_butterworth == 1:
            # Chebyshev Type II, wide band
            sos = cheby2(4, 40, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass', output='sos')
            predictions = sosfiltfilt(sos, np.double(predictions))
            labels = sosfiltfilt(sos, np.double(labels))
            print("\nrppg-toolbox Filter: Chebyshev II(4), fl=0.6, fh=3.3")

        elif evaluate_butterworth == 2:
            # Welch method: wide butter(4) bandpass + Welch PSD for HR extraction
            # Bandpass same as SQI script: [0.5, 8.0] Hz for signal preservation
            sos = butter(4, [0.5 / fs * 2, 8.0 / fs * 2], btype='bandpass', output='sos')
            predictions = sosfiltfilt(sos, np.double(predictions))
            labels = sosfiltfilt(sos, np.double(labels))
            print("\nrppg-toolbox Filter: Butter(4) wide [0.5,8.0] + Welch PSD")

    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        if evaluate_butterworth == 2:
            # Use Welch PSD to find HR (smoother, more noise-robust)
            hr_pred = _calculate_welch_hr(predictions, fs=fs)
            hr_label = _calculate_welch_hr(labels, fs=fs)
        else:
            # Use periodogram (original toolbox method)
            hr_pred = _calculate_fft_hr(predictions, fs=fs)
            hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    print(f"See Me: [HR] GT={hr_label:.1f} BPM, Pred={hr_pred:.1f} BPM, diff={abs(hr_pred - hr_label):.1f}")
    return hr_label, hr_pred, SNR, macc