"""The dataloader for Aya's custom rPPG dataset.

=== Key Modifications (v2) ===
1. SUBJECT-LEVEL SPLIT: Uses explicit subject ID lists instead of BEGIN/END ratio.
   Controlled via YAML: TRAIN_SUBJECTS, TEST_SUBJECTS
2. CONDITION FILTERING: Filters sessions by distance, motion, lux via config lists.
   Controlled in this file: TRAIN_CONDITIONS / TEST_CONDITIONS dicts.
3. MOTION-LEVEL METADATA: Each preprocessed chunk stores its motion_level
   (e.g., "Stationary", "Talking") in a side-car .npy file for granular evaluation.

Dataset Structure:
    DataRoot/
    |-- 0/                              <-- Subject folder (0~10)
    |   |-- vid_1m_30lux_Stationary_45deg_useiPhone/
    |   |   |-- 1709812345000.png       <-- Frame (filename = timestamp ms)
    |   |-- gt_1m_30lux_Stationary_45deg_useiPhone/
    |   |   |-- pulse_data.csv          <-- Columns: PC_Timestamp_ms, Signal_Value
    |-- 1/
    |-- ...
    |-- 10/
"""
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

# ============================================================================
#  CONFIGURATION: Edit these to control which sessions are loaded
# ============================================================================

# --- Subject Split ---
TRAIN_SUBJECTS = [ 1, 2, 3, 6, 8, 9]
TEST_SUBJECTS  = [4, 5, 7]

# --- Train Conditions: which (distance, motion, lux) combos to include for training ---
# Set a list to filter, or None / empty list to include ALL values for that dimension.
TRAIN_CONDITIONS = {
    "distances": [1, 2, 3,4,5],               # e.g., [1, 2] for 1m and 2m
    "motions":   None,         # None = all motions (Stationary, Talking, etc.)
    "lux":       [0,30,70,130],                  # None = all lux levels
}

# --- Test Conditions: which combos to include for testing ---
# Typically broader to evaluate across all conditions
TEST_CONDITIONS = {
    "distances": [3],
    "motions":   ['Stationary'],                  # None = all motions
    "lux":       [70],
}


def _parse_folder_name(folder_name):
    """Parse a vid/gt folder name into its condition components.

    Example: 'vid_1m_30lux_Stationary_45deg_useiPhone'
    Returns: {'distance': 1, 'lux': 30, 'motion': 'Stationary', 'angle': '45deg', 'device': 'useiPhone'}
    """
    # Remove vid_ or gt_ prefix
    if folder_name.startswith("vid_"):
        core = folder_name[4:]
    elif folder_name.startswith("gt_"):
        core = folder_name[3:]
    else:
        core = folder_name

    parts = core.split("_")
    result = {}
    try:
        result["distance"] = int(parts[0].replace("m", ""))   # "1m" -> 1
        result["lux"] = int(parts[1].replace("lux", ""))       # "30lux" -> 30
        result["motion"] = parts[2]                             # "Stationary"
        result["angle"] = parts[3] if len(parts) > 3 else ""   # "45deg"
        result["device"] = parts[4] if len(parts) > 4 else ""  # "useiPhone"
    except (IndexError, ValueError):
        result["distance"] = -1
        result["lux"] = -1
        result["motion"] = "Unknown"
        result["angle"] = ""
        result["device"] = ""
    return result


def _matches_conditions(parsed, conditions):
    """Check if a parsed folder matches the given condition filters.

    Args:
        parsed: dict from _parse_folder_name
        conditions: dict with keys 'distances', 'motions', 'lux' (each a list or None)
    Returns:
        True if the folder matches ALL non-None filters.
    """
    if conditions.get("distances"):
        if parsed["distance"] not in conditions["distances"]:
            return False
    if conditions.get("motions"):
        if parsed["motion"] not in conditions["motions"]:
            return False
    if conditions.get("lux"):
        if parsed["lux"] not in conditions["lux"]:
            return False
    return True


class AyaLoader(BaseLoader):
    """The data loader for Aya's custom rPPG dataset (v2 with subject split + condition filtering)."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes the Aya dataloader.

        Args:
            name(str): "train", "valid", or "test" — determines which subject/condition list to use.
            data_path(str): path to the root folder containing subject directories (0~10).
            config_data(CfgNode): data settings (ref: config.py).
        """
        # Store split name BEFORE calling super().__init__ which triggers preprocessing
        self.split_name = name  # "train", "valid", or "test"

        # Metadata: maps chunk filename (str) -> motion_level (str)
        # Populated during preprocessing, saved/loaded as JSON sidecar
        self.chunk_motion_map = {}

        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Scan dataset root and return all valid vid/gt pairs with metadata.

        Filters by SUBJECT list and CONDITION list based on self.split_name.
        """
        # Select subjects and conditions based on split
        if self.split_name in ("train",):
            allowed_subjects = TRAIN_SUBJECTS
            conditions = TRAIN_CONDITIONS
        else:  # "valid" or "test"
            allowed_subjects = TEST_SUBJECTS
            conditions = TEST_CONDITIONS

        data_dirs = []
        subject_folders = sorted(
            [d for d in os.listdir(data_path)
             if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()],
            key=lambda x: int(x)
        )
        if not subject_folders:
            raise ValueError(self.dataset_name + " data paths empty!")

        trial_counter = 0
        for subj_folder in subject_folders:
            subject = int(subj_folder)

            # --- Subject filter ---
            if subject not in allowed_subjects:
                continue

            subj_path = os.path.join(data_path, subj_folder)
            vid_folders = sorted(glob.glob(os.path.join(subj_path, "vid_*")))

            for vid_folder in vid_folders:
                vid_name = os.path.basename(vid_folder)
                parsed = _parse_folder_name(vid_name)

                # --- Condition filter ---
                if not _matches_conditions(parsed, conditions):
                    continue

                gt_name = "gt_" + vid_name[4:]
                gt_folder = os.path.join(subj_path, gt_name)

                if os.path.isdir(gt_folder):
                    data_dirs.append({
                        "index": trial_counter,
                        "path": vid_folder,
                        "subject": subject,
                        "gt_path": gt_folder,
                        "motion_level": parsed["motion"],   # <-- NEW: store motion
                        "distance": parsed["distance"],
                        "lux": parsed["lux"],
                    })
                    trial_counter += 1

        if not data_dirs:
            raise ValueError(
                f"{self.dataset_name} [{self.split_name}] no valid vid/gt pairs found! "
                f"Subjects={allowed_subjects}, Conditions={conditions}"
            )

        print(f"\n[AyaLoader] Split='{self.split_name}', "
              f"Subjects={allowed_subjects}, "
              f"Sessions found={len(data_dirs)}")
        for d in data_dirs:
            print(f"  idx={d['index']} subj={d['subject']} "
                  f"motion={d['motion_level']} dist={d['distance']}m "
                  f"lux={d['lux']} -> {os.path.basename(d['path'])}")
        print()

        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Override: subject filtering is already done in get_raw_data.

        BEGIN/END are ignored — the split is fully controlled by subject lists.
        Return all data_dirs as-is.
        """
        return data_dirs

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=4):
        """Override: limit parallel processes."""
        return super().multi_process_manager(data_dirs, config_preprocess, multi_process_quota=4)

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocess one session: read frames, align GT, crop face, chunk, save.

        Also saves motion_level metadata for each chunk.
        """
        saved_filename = data_dirs[i]['index']
        vid_path = data_dirs[i]['path']
        gt_path = data_dirs[i]['gt_path']
        motion_level = data_dirs[i]['motion_level']
        print(f"\n[{i + 1}/{len(data_dirs)}] Processing: {os.path.basename(vid_path)} "
              f"(motion={motion_level})")

        # ---- Step 1: Read & trim ground truth ----
        gt_csv = os.path.join(gt_path, "pulse_data.csv")
        if not os.path.exists(gt_csv):
            print(f"[WARN] GT not found, skipping: {gt_csv}")
            file_list_dict[i] = []
            return

        gt_df = pd.read_csv(gt_csv)
        gt_df = gt_df[["PC_Timestamp_ms", "Signal_Value"]].dropna()
        gt_df = gt_df.iloc[100:-100].reset_index(drop=True)
        if len(gt_df) < 30:
            print(f"[WARN] GT too short after trim, skipping: {gt_csv}")
            file_list_dict[i] = []
            return

        gt_timestamps = gt_df["PC_Timestamp_ms"].values.astype(np.float64)
        gt_signal = gt_df["Signal_Value"].values.astype(np.float64)

        # ---- Step 2: Get filenames & timestamps (no pixel loading yet) ----
        all_png = sorted(glob.glob(vid_path + os.sep + '*.png'))
        frame_ts = np.array([
            float(os.path.splitext(os.path.basename(p))[0]) for p in all_png
        ])

        # ---- Step 3: Filter to GT time range ----
        gt_start, gt_end = gt_timestamps[0], gt_timestamps[-1]
        mask = (frame_ts >= gt_start) & (frame_ts <= gt_end)
        selected_png = [p for p, m in zip(all_png, mask) if m]
        frame_ts = frame_ts[mask]

        if len(selected_png) < 30:
            print(f"[WARN] Too few frames after alignment, skipping: {vid_path}")
            file_list_dict[i] = []
            return

        # ---- Step 4: Align GT signal to frame timestamps ----
        indices = np.searchsorted(gt_timestamps, frame_ts, side='left')
        indices = np.clip(indices, 0, len(gt_timestamps) - 1)
        bvps = gt_signal[indices]

        # ---- Step 5: Read frames (memory efficient pre-resize) ----
        if 'None' in config_preprocess.DATA_AUG:
            frames = self.read_video_list(selected_png, resize_w=640, resize_h=360)
        elif 'Motion' in config_preprocess.DATA_AUG:
            frames = self.read_npy_video(
                glob.glob(os.path.join(vid_path, '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG for {self.dataset_name}!')

        # ---- Step 6: Preprocess (face crop, normalize, chunk) & Save ----
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename)

        # ---- Step 7: Save motion_level metadata for each chunk ----
        # Maps each chunk's input filename -> motion level
        motion_map = {}
        for inp_path in input_name_list:
            chunk_basename = os.path.basename(inp_path)  # e.g., "3_input0.npy"
            motion_map[chunk_basename] = motion_level

        # Save as JSON sidecar next to the cached data
        meta_path = os.path.join(
            self.cached_path,
            f"{saved_filename}_motion_meta.json"
        )
        with open(meta_path, 'w') as f:
            json.dump(motion_map, f)

        file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        """Override: load preprocessed data AND motion metadata."""
        super().load_preprocessed_data()

        # Load all motion metadata JSONs
        self.chunk_motion_map = {}
        meta_files = glob.glob(os.path.join(self.cached_path, "*_motion_meta.json"))
        for mf in meta_files:
            with open(mf, 'r') as f:
                self.chunk_motion_map.update(json.load(f))

        if self.chunk_motion_map:
            # Count per motion level
            motion_counts = {}
            for ml in self.chunk_motion_map.values():
                motion_counts[ml] = motion_counts.get(ml, 0) + 1
            print(f"[AyaLoader] Motion metadata loaded: {motion_counts}")

    def get_motion_level(self, index):
        """Get motion level for a given dataset index.

        Useful for granular test evaluation.
        Returns: str like 'Stationary', 'Talking', etc. or 'Unknown'
        """
        item_path = self.inputs[index]
        chunk_basename = os.path.basename(item_path)
        return self.chunk_motion_map.get(chunk_basename, "Unknown")

    # ---- Utility methods (unchanged) ----

    @staticmethod
    def read_video(video_file):
        """Reads all .png frames from a directory, returns (T, H, W, 3)."""
        frames = []
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    def read_video_list(self, png_list, resize_w=640, resize_h=360):
        """Read selected frames with immediate resize to save memory."""
        frames = []
        for png_path in png_list:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads BVP signal from pulse_data.csv."""
        df = pd.read_csv(bvp_file)
        return df["Signal_Value"].values.astype(np.float64)