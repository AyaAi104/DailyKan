"""The dataloader for Aya's custom rPPG dataset.

Dataset Structure:
    DataRoot/
    |-- 1/                              <-- Subject folder (1~10)
    |   |-- vid_1m_30lux_.../           <-- Video frames folder
    |   |   |-- 1709812345000.png       <-- Frame (filename = timestamp ms)
    |   |   |-- ...
    |   |-- gt_1m_30lux_.../            <-- Ground truth folder
    |   |   |-- pulse_data.csv          <-- Columns: PC_Timestamp_ms, Signal_Value
    |-- 2/
    |-- ...
    |-- 10/

Synchronization:
    Frame filenames ARE timestamps (ms). The GT CSV column PC_Timestamp_ms
    is used to align frames with Signal_Value via nearest-neighbor matching.
    First and last 100 GT rows are trimmed to avoid sensor edge artifacts.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

distance = 1
distances = [4]
motion = "Stationary"
class AyaLoader(BaseLoader):
    """The data loader for Aya's custom rPPG dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes the Aya dataloader.
            Args:
                data_path(str): path to the root folder containing subject directories (1~10).
                name(str): name of the dataloader.
                config_data(CfgNode): data settings (ref: config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path (for Aya dataset).

        Scans each subject folder for vid_*/gt_* pairs.
        Each pair becomes one entry with a unique index for caching.

        """
        data_dirs = []
        # Find all subject folders (digits only: 1, 2, ... 10)
        subject_folders = sorted(
            [d for d in os.listdir(data_path)
             if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()],
            key=lambda x: int(x)
        )
        if not subject_folders:
            raise ValueError(self.dataset_name + " data paths empty ")

        trial_counter = 0  # global unique index
        for subj_folder in subject_folders:
            subject = int(subj_folder)
            subj_path = os.path.join(data_path, subj_folder)
            vid_folders = []
            # Find all vid_ folders inside this subject
            if distances:
                # vid_folders = sorted(glob.glob(os.path.join(subj_path, f"vid_{distance}m_*")))
                for d in distances:
                    pattern = os.path.join(subj_path, f"vid_{d}m_*{motion}*")
                    vid_folders += glob.glob(pattern)
            else:
                vid_folders = sorted(glob.glob(os.path.join(subj_path, "vid_*")))
            for vid_folder in vid_folders:
                vid_name = os.path.basename(vid_folder)
                gt_name = "gt_" + vid_name[4:]       # vid_1m_... map -> gt_1m_...
                gt_folder = os.path.join(subj_path, gt_name)

                # Only add if corresponding gt folder exists
                if os.path.isdir(gt_folder):
                    data_dirs.append({
                        "index": trial_counter,
                        "path": vid_folder,
                        "subject": subject,
                        "gt_path": gt_folder
                    })
                    trial_counter += 1

        if not data_dirs:
            raise ValueError(self.dataset_name + " no valid vid/gt pairs found!")
        return data_dirs

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=2):
        """Override: limit to 2 parallel processes to avoid OOM.

        """
        return super().multi_process_manager(data_dirs, config_preprocess, multi_process_quota=4)
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split by subject to prevent data leakage.

        """
        if begin == 0 and end == 1:
            return data_dirs

        # Group by subject
        data_info = {}
        for data in data_dirs:
            subject = data['subject']
            if subject not in data_info:
                data_info[subject] = []
            data_info[subject].append(data)

        subj_list = sorted(data_info.keys())
        num_subjs = len(subj_list)
        subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        data_dirs_new = []
        for i in subj_range:
            data_dirs_new += data_info[subj_list[i]]
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Memory-efficient preprocessing: filter filenames first, then read+resize one by one.

        """
        saved_filename = data_dirs[i]['index']
        vid_path = data_dirs[i]['path']
        gt_path = data_dirs[i]['gt_path']
        print(f"\n[{i + 1}/{len(data_dirs)}] Processing: {os.path.basename(vid_path)}")

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

        # ---- Step 2: Get filenames & timestamps only (no pixel loading) ----
        all_png = sorted(glob.glob(vid_path + os.sep + '*.png'))
        frame_ts = np.array([
            float(os.path.splitext(os.path.basename(p))[0]) for p in all_png
        ])

        # ---- Step 3: Filter to GT time range (still just filenames) ----
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
        # indices = [ First Match, +1, +2...]
        indices = np.clip(indices, 0, len(gt_timestamps) - 1)
        bvps = gt_signal[indices]

        # If Use Linear Interpolation:
        #bvps = np.interp(frame_ts, gt_timestamps, gt_signal)

        target_length = len(selected_png)

        # optional
        #bvps = BaseLoader.resample_ppg(bvps, target_length)

        # ---- Step 5: Read frames one-by-one (memory efficient) ----

        if 'None' in config_preprocess.DATA_AUG:
            frames = self.read_video_list(selected_png,
                resize_w=640,
                resize_h=360)
        elif 'Motion' in config_preprocess.DATA_AUG:
            frames = self.read_npy_video(
                glob.glob(os.path.join(vid_path, '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG for {self.dataset_name}!')

        # ---- Step 6: Preprocess & Save ----
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

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


    def read_video_list(self,png_list, resize_w=72, resize_h=72):
        """Read selected frames, resize immediately to save memory.

        instanly resize。
        4000 × 72×72×3 = ~56MB（
        """
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