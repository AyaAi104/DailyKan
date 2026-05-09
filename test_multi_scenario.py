"""
Multi-Scenario Test Script for Aya rPPG Dataset
================================================
Runs a trained model across multiple (distance, lux, motion) scenarios in ONE execution,
reporting per-scenario metrics individually.

Usage:
    python test_multi_scenario.py --config_file <your_yaml> [--model_path <path_to_pth>]
    python test_multi_scenario.py --config_file configs/train_configs/Daily_Daily_Daily_TSCAN_try.yaml --model_path ./runs/exp/Aya_SizeW72_SizeH72_ClipLength300_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendINSIGHT_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/PreTrainedModels/Aya_Aya_TSCAN_1m_v2_Epoch11.pth --subjects 4 5 7 --plot
    python test_multi_scenario.py --config_file configs/train_configs/Daily_Daily_Daily_DEEPPHYS_split.yaml --model_path ./runs/deepphys/Aya_SizeW72_SizeH72_ClipLength300_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendINSIGHT_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/PreTrainedModels/Aya_Aya_DEEPPHYS_1m_v2_Epoch9.pth --subjects 4 5 7 --plot --plot_dir ./runs/deepphys/waveforms
    python test_multi_scenario.py --config_file ./configs/train_configs/PURE_PURE_Daily_TSCAN_BASIC.yaml --model_path runs/pure\PURE_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse\PreTrainedModels\PURE_PURE_Aya_tscan_Epoch29.pth --subjects 4 5 7 --plot --plot_dir ./runs/pure/waveforms
python test_multi_scenario.py --config_file ./configs/train_configs/PURE_PURE_AYA_RHYTHMFORMER.yaml --model_path ./runs/rhythm/PURE_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/PreTrainedModels/PURE_AYA_RhythmFormer_Epoch16.pth --subjects 3 4 5 6 7 8 9 --plot --plot_dir ./runs/rhythm/waveforms
The SCENARIOS list below controls which combinations are tested.
Edit it to match your needs.
"""

import argparse
import copy
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- rPPG Toolbox imports (adjust paths if needed) ----
from config import get_config
from dataset.data_loader import AyaLoader as AyaLoaderModule
from evaluation.metrics import calculate_metrics

# ============================================================================
#  USER CONFIG: Define your test scenarios here
# ============================================================================

# Test subjects (same as AyaLoader.TEST_SUBJECTS default)
#TEST_SUBJECTS = [4, 5, 7]
TEST_SUBJECTS = [3,4,5,6, 7,8,9]

# Each scenario is a dict with keys: distance, lux, motion
# The script will iterate over ALL scenarios and test each one independently.
#
# Your dataset has 5 distances (1~5m) and these scenarios per distance:
#   70lux: Stationary, Nodding, Shaking, Talking
#   30lux: Stationary
# = 5 scenarios × 5 distances = 25 test runs
SCENARIOS = []

# Auto-generate: 5 distances × 5 conditions
for dist in [1,2,3,4,5]:
    # 70 lux conditions
    for motion in ["Stationary","Shaking","Nodding","Talking"]:
    #for motion in ["Nodding"]:
        SCENARIOS.append({"distance": dist, "lux": 70, "motion": motion})
    # 30 lux condition
    # 30 lux condition
    SCENARIOS.append({"distance": dist, "lux": 30, "motion": "Stationary"})
    SCENARIOS.append({"distance": dist, "lux": 130, "motion": "Stationary"})
    SCENARIOS.append({"distance": dist, "lux": 0, "motion": "Stationary"})

# Or you can manually specify a subset, e.g.:
# SCENARIOS = [
#     {"distance": 1, "lux": 70, "motion": "Stationary"},
#     {"distance": 3, "lux": 30, "motion": "Stationary"},
#     {"distance": 5, "lux": 70, "motion": "Nodding"},
# ]

# ============================================================================
#  Reproducibility
# ============================================================================
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def patch_aya_conditions(distances, motions, lux_values, subjects):
    """Monkey-patch AyaLoader module globals so the next AyaLoader('test', ...)
    picks up the desired scenario filters."""
    AyaLoaderModule.TEST_SUBJECTS = subjects
    AyaLoaderModule.TEST_CONDITIONS = {
        "distances": distances if isinstance(distances, list) else [distances],
        "motions": motions if isinstance(motions, list) else [motions],
        "lux": lux_values if isinstance(lux_values, list) else [lux_values],
    }


def build_test_loader(config, scenario, subjects):
    """Build an AyaLoader + DataLoader for one scenario."""
    patch_aya_conditions(
        distances=[scenario["distance"]],
        motions=[scenario["motion"]],
        lux_values=[scenario["lux"]],
        subjects=subjects,
    )

    test_data = AyaLoaderModule.AyaLoader(
        name="test",
        data_path=config.TEST.DATA.DATA_PATH,
        config_data=config.TEST.DATA,
        device=config.DEVICE,
    )

    test_loader = DataLoader(
        dataset=test_data,
        num_workers=4,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=general_generator,
    )
    return test_loader


def load_model(config, model_path=None):
    """Instantiate and load the model based on config.MODEL.NAME.

    Returns the model on the correct device, in eval mode.
    Supports: PhysNet, TSCAN, DeepPhys, EfficientPhys, PhysFormer, iBVPNet, etc.
    Add more branches as needQed.
    """
    device = torch.device(config.DEVICE)
    model_name = config.MODEL.NAME
    model_name_lower = model_name.lower()  # case-insensitive matching

    if model_name_lower == "physnet":
        from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM)
    elif model_name_lower == "tscan":
        from neural_methods.model.TS_CAN import TSCAN
        model = TSCAN(
            frame_depth=config.MODEL.TSCAN.FRAME_DEPTH,
            img_size=config.TEST.DATA.PREPROCESS.RESIZE.H)
    elif model_name_lower == "deepphys":
        from neural_methods.model.DeepPhys import DeepPhys
        #model = DeepPhys()
        model = DeepPhys(img_size=config.TEST.DATA.PREPROCESS.RESIZE.H)
    elif model_name_lower == "efficientphys":
        from neural_methods.model.EfficientPhys import EfficientPhys
        model = EfficientPhys(
            frame_depth=config.MODEL.EFFICIENTPHYS.FRAME_DEPTH,
            img_size=config.TEST.DATA.PREPROCESS.RESIZE.H)
    elif model_name_lower in ("physformer", "physformer_small"):
        from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
        model = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(
                config.TEST.DATA.PREPROCESS.RESIZE.H,
                config.TEST.DATA.PREPROCESS.RESIZE.W,
            ),
            patches=(4, 4, 4),
            dim=96,
            ff_dim=144,
            num_heads=4,
            num_layers=12,
            theta=0.7,
        )
    elif model_name_lower == "ibvpnet":
        from neural_methods.model.iBVPNet import iBVPNet
        model = iBVPNet(frames=config.MODEL.IBVPNET.FRAME_NUM)
    elif model_name_lower == "rhythmformer":
        from neural_methods.model.RhythmFormer import RhythmFormer
        model = RhythmFormer()
    elif model_name_lower == "factorizephys":
        # Build md_config dict exactly like FactorizePhysTrainer does
        md_config = {
            "FRAME_NUM": config.MODEL.FactorizePhys.FRAME_NUM,
            "MD_TYPE": config.MODEL.FactorizePhys.MD_TYPE,
            "MD_FSAM": config.MODEL.FactorizePhys.MD_FSAM,
            "MD_TRANSFORM": config.MODEL.FactorizePhys.MD_TRANSFORM,
            "MD_S": config.MODEL.FactorizePhys.MD_S,
            "MD_R": config.MODEL.FactorizePhys.MD_R,
            "MD_STEPS": config.MODEL.FactorizePhys.MD_STEPS,
            "MD_INFERENCE": config.MODEL.FactorizePhys.MD_INFERENCE,
            "MD_RESIDUAL": config.MODEL.FactorizePhys.MD_RESIDUAL,
        }
        frames = config.MODEL.FactorizePhys.FRAME_NUM
        in_channels = config.MODEL.FactorizePhys.CHANNELS
        model_type = config.MODEL.FactorizePhys.TYPE.lower()

        if model_type == "standard":
            from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
            model = FactorizePhys(
                frames=frames, md_config=md_config,
                in_channels=in_channels, dropout=config.MODEL.DROP_RATE,
                device=device)
        elif model_type == "big":
            from neural_methods.model.FactorizePhys.FactorizePhysBig import FactorizePhysBig
            model = FactorizePhysBig(
                frames=frames, md_config=md_config,
                in_channels=in_channels, dropout=config.MODEL.DROP_RATE,
                device=device)
        else:
            raise ValueError(f"Unknown FactorizePhys TYPE: {model_type}")

    else:
        raise ValueError(f"Unsupported MODEL.NAME: {model_name}. "
                         f"Add a branch in load_model() for your model.")

    # Load weights
    path = model_path or config.INFERENCE.MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"Loading model weights from: {path}")
    state_dict = torch.load(path, map_location=device)
    # Handle models saved with DataParallel (keys prefixed with "module.")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, test_loader, config):
    """Run inference on one DataLoader. Returns (predictions_dict, labels_dict).

    The dict structure matches the toolbox convention:
        predictions[subj_index][sort_index] = pred_tensor
        labels[subj_index][sort_index]      = label_tensor
    """
    device = torch.device(config.DEVICE)
    model_name = config.MODEL.NAME
    model_name_lower = model_name.lower()  # case-insensitive matching
    chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
    num_gpu = config.NUM_OF_GPU_TRAIN
    base_len = max(num_gpu, 1)

    predictions = dict()
    labels = dict()

    from tqdm import tqdm

    with torch.no_grad():
        for _, test_batch in enumerate(tqdm(test_loader, ncols=80, desc="Inference")):
            batch_size = test_batch[0].shape[0]
            data = test_batch[0].to(device)
            label = test_batch[1].to(device)

            # --- Model-specific forward pass ---
            if model_name_lower in ("tscan", "deepphys"):
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                label = label.view(-1, 1)
                data = data[:(N * D) // base_len * base_len]
                label = label[:(N * D) // base_len * base_len]
                pred = model(data)

            elif model_name_lower == "efficientphys":
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                label = label.view(-1, 1)
                data = data[:(N * D) // base_len * base_len]
                label = label[:(N * D) // base_len * base_len]
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(
                    max(num_gpu, 1), 1, 1, 1)
                data = torch.cat((data, last_frame), 0)
                pred = model(data)

            elif model_name_lower == "physnet":
                pred, _, _, _ = model(data)

            elif model_name_lower in ("physformer", "physformer_small"):
                pred, _, _, _ = model(data, 0.0)  # gra_sharp=0.0 at test time

            elif model_name_lower == "ibvpnet":
                pred = model(data)
            elif model_name_lower == "rhythmformer":
                pred = model(data)
            elif model_name_lower == "factorizephys":

                if len(label.shape) > 2:
                    label = label[..., 0]
                label = (label - torch.mean(label)) / (torch.std(label) + 1e-8)

                # (b)
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(
                    1, 1, max(num_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # (c)
                out = model(data)
                if isinstance(out, tuple):
                    pred = out[0]
                else:
                    pred = out

                # Normal
                pred = (pred - torch.mean(pred)) / (torch.std(pred) + 1e-8)

            else:
                # Generic fallback
                pred = model(data)

            pred = pred.cpu()
            label = label.cpu()

            # --- Collect per-sample predictions ---
            for idx in range(batch_size):
                subj_index = test_batch[2][idx]
                sort_index = int(test_batch[3][idx])

                if subj_index not in predictions:
                    predictions[subj_index] = dict()
                    labels[subj_index] = dict()

                if model_name_lower in ("tscan", "deepphys", "efficientphys"):
                    predictions[subj_index][sort_index] = pred[
                        idx * chunk_len:(idx + 1) * chunk_len]
                    labels[subj_index][sort_index] = label[
                        idx * chunk_len:(idx + 1) * chunk_len]
                else:
                    predictions[subj_index][sort_index] = pred[idx]
                    labels[subj_index][sort_index] = label[idx]

    # ========================================================================
    #  DEBUG: Compare Welch PSD vs High-res FFT per subject
    # ========================================================================
    from scipy import signal as sci_signal
    fs = config.TEST.DATA.FS

    # ========================================================================
    print()
    return predictions, labels


def scenario_label(s):
    """Human-readable label for a scenario dict."""
    return f"{s['distance']}m_{s['lux']}lux_{s['motion']}"


def plot_waveforms(predictions, labels_dict, config, scenario_tag, out_dir):
    """Plot using the EXACT same pipeline as toolbox's calculate_metric_per_video.

    3 columns per subject:
      1. Time-domain waveform AFTER cumsum+detrend+bandpass (what toolbox evaluates)
      2. FFT spectrum with _next_power_of_2 nfft (what toolbox uses to find HR)
      3. Per-chunk HR scatter (pred vs GT, one dot per chunk)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evaluation.post_process import _detrend, _next_power_of_2, _calculate_fft_hr
    from scipy.signal import butter, filtfilt, periodogram

    fs = config.TEST.DATA.FS
    diff_flag = (config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized")

    subj_keys = sorted(predictions.keys(), key=lambda x: str(x))
    n_subj = len(subj_keys)
    if n_subj == 0:
        return

    fig, axes = plt.subplots(n_subj, 3, figsize=(36, 6 * n_subj),
                             squeeze=False, gridspec_kw={"width_ratios": [5, 2, 1.5]})
    fig.suptitle(f"Toolbox-Exact Evaluation — {scenario_tag}",
                 fontsize=16, fontweight="bold", y=1.01)

    for row, subj in enumerate(subj_keys):
        sort_keys = sorted(predictions[subj].keys(), key=lambda x: int(x))
        pred_concat = np.concatenate(
            [predictions[subj][k].numpy().flatten() for k in sort_keys])
        label_concat = np.concatenate(
            [labels_dict[subj][k].numpy().flatten() for k in sort_keys])

        # ============================================================
        #  EXACT toolbox pipeline (from calculate_metric_per_video)
        # ============================================================
        if diff_flag:
            pred_proc = _detrend(np.cumsum(pred_concat), 100)
            gt_proc = _detrend(np.cumsum(label_concat), 100)
        else:
            pred_proc = _detrend(pred_concat, 100)
            gt_proc = _detrend(label_concat, 100)
        #$1
        [b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        print("Plot Filter: I use Butterworth, fl is 0.6, fh is 3.3")
        pred_proc = filtfilt(b, a, np.double(pred_proc))
        gt_proc = filtfilt(b, a, np.double(gt_proc))

        # FFT — exact same as _calculate_fft_hr
        N = _next_power_of_2(len(pred_proc))
        f_pred, pxx_pred = periodogram(pred_proc, fs=fs, nfft=N, detrend=False)
        f_gt, pxx_gt = periodogram(gt_proc, fs=fs, nfft=N, detrend=False)
        # $2
        fmask = (f_pred >= 0.8) & (f_pred <= 1.667)
        print("Find HR: fl is 0.8(48), fh is 1.667(100)")
        hr_pred = f_pred[fmask][np.argmax(pxx_pred[fmask])] * 60
        hr_gt = f_gt[fmask][np.argmax(pxx_gt[fmask])] * 60

        # ============================================================
        #  Panel 1: Processed time-domain waveform
        # ============================================================
        ax1 = axes[row, 0]
        t = np.arange(len(gt_proc)) / fs
        # Normalize for visual overlay
        gt_plot = (gt_proc - np.mean(gt_proc)) / (np.std(gt_proc) + 1e-8)
        pred_plot = (pred_proc - np.mean(pred_proc)) / (np.std(pred_proc) + 1e-8)
        ax1.plot(t, gt_plot, color="#2196F3", lw=0.7, alpha=0.85, label="GT (processed)")
        ax1.plot(t, pred_plot, color="#F44336", lw=0.7, alpha=0.75, label="Pred (processed)")
        ax1.set_title(f"Subject {subj} — After cumsum→detrend→bandpass "
                      f"({len(sort_keys)} chunks, {len(pred_proc)} samples)",
                      fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Normalized Amplitude")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        # Chunk boundaries
        cum = 0
        for k in sort_keys[:-1]:
            cum += len(predictions[subj][k].numpy().flatten())
            ax1.axvline(cum / fs, color="#BDBDBD", ls=":", lw=0.5)

        # ============================================================
        #  Panel 2: FFT spectrum (exact toolbox method)
        # ============================================================
        ax2 = axes[row, 1]
        bpm_axis = f_pred * 60  # Hz to BPM
        bpm_mask = (bpm_axis >= 30) & (bpm_axis <= 200)

        ax2.plot(bpm_axis[bpm_mask], pxx_gt[bpm_mask],
                 color="#2196F3", lw=1.0, alpha=0.85, label="GT")
        ax2.plot(bpm_axis[bpm_mask], pxx_pred[bpm_mask],
                 color="#F44336", lw=1.0, alpha=0.75, label="Pred")

        # Mark peaks
        ax2.axvline(hr_gt, color="#2196F3", ls="--", lw=1.2, alpha=0.7)
        ax2.axvline(hr_pred, color="#F44336", ls="--", lw=1.2, alpha=0.7)
        ymax = max(np.max(pxx_gt[bpm_mask]), np.max(pxx_pred[bpm_mask]))
        ax2.text(hr_gt, ymax * 0.95, f" GT:{hr_gt:.1f}",
                 fontsize=9, color="#2196F3", fontweight="bold")
        ax2.text(hr_pred, ymax * 0.80, f" Pred:{hr_pred:.1f}",
                 fontsize=9, color="#F44336", fontweight="bold")

        ax2.set_title(f"FFT (nfft={N}, toolbox exact)\n"
                      f"MAE={abs(hr_pred - hr_gt):.1f} BPM",
                      fontsize=11, fontweight="bold")
        ax2.set_xlabel("BPM")
        ax2.set_ylabel("Power")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ============================================================
        #  Panel 3: Per-chunk HR (pred vs GT)
        # ============================================================
        ax3 = axes[row, 2]
        chunk_hr_pred = []
        chunk_hr_gt = []
        for k in sort_keys:
            p = predictions[subj][k].numpy().flatten()
            l = labels_dict[subj][k].numpy().flatten()
            try:
                if diff_flag:
                    p = _detrend(np.cumsum(p), 100)
                    l = _detrend(np.cumsum(l), 100)
                else:
                    p = _detrend(p, 100)
                    l = _detrend(l, 100)
                if len(p) > 9:
                    [b2, a2] = butter(1, [0.8/fs*2, 3.3/fs*2], btype='bandpass')
                    p = filtfilt(b2, a2, np.double(p))
                    l = filtfilt(b2, a2, np.double(l))
                chunk_hr_pred.append(_calculate_fft_hr(p, fs=fs))
                chunk_hr_gt.append(_calculate_fft_hr(l, fs=fs))
            except Exception:
                pass

        chunk_hr_pred = np.array(chunk_hr_pred)
        chunk_hr_gt = np.array(chunk_hr_gt)
        if len(chunk_hr_pred) > 0:
            ax3.scatter(chunk_hr_gt, chunk_hr_pred, c="#9C27B0", s=30, alpha=0.7)
            # Identity line
            lo = min(np.min(chunk_hr_gt), np.min(chunk_hr_pred)) - 5
            hi = max(np.max(chunk_hr_gt), np.max(chunk_hr_pred)) + 5
            ax3.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax3.set_xlim(lo, hi)
            ax3.set_ylim(lo, hi)
            chunk_mae = np.mean(np.abs(chunk_hr_pred - chunk_hr_gt))
            ax3.set_title(f"Per-Chunk HR\nMAE={chunk_mae:.1f} BPM",
                          fontsize=11, fontweight="bold")
        else:
            ax3.set_title("Per-Chunk HR\n(no data)", fontsize=11)
        ax3.set_xlabel("GT HR (BPM)")
        ax3.set_ylabel("Pred HR (BPM)")
        ax3.set_aspect("equal")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    os.makedirs(out_dir, exist_ok=True)
    safe_tag = scenario_tag.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(out_dir, f"waveform_{safe_tag}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Waveform plot saved: {save_path}")


# ============================================================================
#  MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-scenario test for Aya rPPG dataset")
    parser.add_argument("--config_file", required=True, type=str,
                        help="Path to the YAML config (should have TEST section for Aya)")
    parser.add_argument("--model_path", default=None, type=str,
                        help="Override INFERENCE.MODEL_PATH in config")
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Override test subjects (e.g., --subjects 4 5 7)")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Save predicted vs GT waveform plots for each scenario")
    parser.add_argument("--plot_dir", default=None, type=str,
                        help="Directory to save waveform plots (default: runs/exp/waveforms)")
    args = parser.parse_args()

    # Load config
    config = get_config(args)

    # Override subjects if given
    subjects = args.subjects if args.subjects else TEST_SUBJECTS

    # Load model ONCE
    model = load_model(config, args.model_path)
    print(f"\nModel loaded: {config.MODEL.NAME}")
    print(f"Test subjects: {subjects}")
    print(f"Number of scenarios: {len(SCENARIOS)}")
    print("=" * 70)

    # Results accumulator
    all_results = OrderedDict()

    for i, scenario in enumerate(SCENARIOS):
        tag = scenario_label(scenario)
        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{len(SCENARIOS)}] Testing scenario: {tag}")
        print(f"  distance={scenario['distance']}m, "
              f"lux={scenario['lux']}, motion={scenario['motion']}")
        print(f"  subjects={subjects}")
        print(f"{'=' * 70}")

        try:
            test_loader = build_test_loader(config, scenario, subjects)
        except ValueError as e:
            print(f"  [SKIP] No data found for this scenario: {e}")
            all_results[tag] = {"status": "NO_DATA", "error": str(e)}
            continue

        if len(test_loader.dataset) == 0:
            print(f"  [SKIP] Empty dataset for scenario {tag}")
            all_results[tag] = {"status": "EMPTY"}
            continue

        print(f"  Dataset size: {len(test_loader.dataset)} chunks")

        # Run inference
        predictions, labels_dict = run_inference(model, test_loader, config)

        # Calculate and print metrics for this scenario
        print(f"\n  --- Metrics for {tag} ---")
        metric_results = calculate_metrics(predictions, labels_dict, config)

        # Plot waveforms if requested
        if args.plot:
            plot_dir = args.plot_dir or os.path.join(
                config.LOG.PATH if hasattr(config.LOG, "PATH") else "runs/exp",
                "waveforms")
            plot_waveforms(predictions, labels_dict, config, tag, plot_dir)

        all_results[tag] = {
            "status": "OK",
            "num_chunks": len(test_loader.dataset),
            "num_subjects": len(predictions),
            "metrics": metric_results,
        }

    # ---- Summary Table ----
    print("\n" + "=" * 90)
    print("MULTI-SCENARIO TEST SUMMARY")
    print("=" * 90)
    print(f"{'Scenario':<35s} {'Status':<10s} {'Chunks':<8s} {'Subjects':<10s}")
    print("-" * 90)
    for tag, res in all_results.items():
        status = res.get("status", "?")
        chunks = str(res.get("num_chunks", "-"))
        subjs = str(res.get("num_subjects", "-"))
        print(f"{tag:<35s} {status:<10s} {chunks:<8s} {subjs:<10s}")
    print("=" * 90)

    # Optionally save results
    import json
    out_dir = config.LOG.PATH if hasattr(config.LOG, "PATH") else "runs/exp"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multi_scenario_results.json")

    # Convert any non-serializable values
    serializable = {}
    for tag, res in all_results.items():
        entry = {}
        for k, v in res.items():
            if isinstance(v, (dict, str, int, float, list)):
                entry[k] = v
            else:
                entry[k] = str(v)
        serializable[tag] = entry

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()