"""Custom Data Loader for Unsynchronized rPPG Dataset
======================================================
A production-ready BaseLoader subclass for the rPPG-Toolbox that handles
datasets where video frames and ground truth CSV signals have DIFFERENT
sampling rates and start/end times.

Synchronization Pipeline (同步流水线):
    Step A — Trimming    (修剪): Remove unstable first/last 100 frames.
    Step B — Alignment   (对齐): pandas.merge_asof nearest-neighbor matching.
    Step C — Formatting  (格式化): Resize, normalize, chunk, and save as .npy.

Dataset Layout:
    Dataset_Root/
    ├── 1/
    │   ├── vid_<conditions>/       <- PNG frames (filename = timestamp_ms)
    │   │   ├── 1709812345000.png
    │   │   ├── 1709812345033.png
    │   │   └── ... (~4000 frames)
    │   └── gt_<conditions>/        <- Ground truth
    │       └── pulse_data.csv      <- Columns: PC_Timestamp_ms, Signal_Value
    ├── 2/
    └── ... up to 10/

Usage:
    1. Place this file in  dataset/data_loader/CustomLoader.py
    2. Register in dataset/data_loader/__init__.py:
         import dataset.data_loader.CustomLoader
    3. Add to main.py routing (train/valid/test blocks):
         elif config.TRAIN.DATA.DATASET == "CustomRPPG":
             train_loader = data_loader.CustomLoader.CustomLoader
    4. Create a YAML config (see bottom of this file for a template).

Author: rPPG-Toolbox Core Contributor
"""

import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class DailyLoader(BaseLoader):
    """Data loader for unsynchronized rPPG datasets with timestamp-based alignment.

    Inherits from BaseLoader and implements the required interface:
        - get_raw_data(data_path)
        - split_raw_data(data_dirs, begin, end)
        - preprocess_dataset_subprocess(data_dirs, config_preprocess, i, file_list_dict)
        - read_video(video_file)   [static]
        - read_wave(bvp_file)      [static]
    """

    # =====================================================================
    #  Configurable Constants (可配置常量)
    # =====================================================================
    TRIM_HEAD = 100   # Number of frames to discard at the START (起始丢弃帧数)
    TRIM_TAIL = 100   # Number of frames to discard at the END   (末尾丢弃帧数)
    ALIGNMENT_TOLERANCE_MS = 50  # Max gap for merge_asof (ms) (对齐容差)

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes the CustomLoader.

        Args:
            name (str): Name of this dataloader instance ("train"/"valid"/"test").
            data_path (str): Path to the raw dataset root directory.
            config_data (CfgNode): Data configuration node from YAML (ref: config.py).
            device: Device for face detection backend (passed to BaseLoader).

        Expected directory layout under data_path:
            data_path/
            ├── 1/
            │   ├── vid_<...>/   (PNG frames, filename = timestamp_ms.png)
            │   └── gt_<...>/    (pulse_data.csv with PC_Timestamp_ms, Signal_Value)
            ├── 2/
            └── ... up to 10/
        """
        super().__init__(name, data_path, config_data, device)

    # =================================================================
    #  STEP 1: Discover Raw Data (发现原始数据)
    # =================================================================
    def get_raw_data(self, data_path):
        """Scans the dataset root and returns a list of data directory descriptors.

        Each subject folder (named 1–10) may contain MULTIPLE recording sessions.
        Sessions are matched by suffix: vid_1m_XXX pairs with gt_1m_XXX.
        每个受试者文件夹可能包含多个录制会话。通过后缀匹配: vid_1m_XXX 与 gt_1m_XXX 配对。

        Returns:
            List[dict]: Each dict has keys:
                - "index":   str identifier, e.g., "subject_3_130lux_Stationary_45deg"
                - "path":    str, full path to the subject folder
                - "subject": int, subject number for split purposes
                - "vid_folder": str, full path to the vid_* subfolder
                - "gt_folder":  str, full path to the gt_* subfolder
        """
        data_dirs = []

        # List all subject directories (numeric folder names)
        # 列出所有受试者目录（数字文件夹名称）
        subject_folders = sorted(
            [d for d in os.listdir(data_path)
             if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()],
            key=lambda x: int(x)
        )

        if not subject_folders:
            raise ValueError(
                f"No numeric subject directories found under {data_path}. "
                f"Expected folders named 1, 2, ..., 10."
            )

        for subj_name in subject_folders:
            subj_path = os.path.join(data_path, subj_name)
            subj_id = int(subj_name)

            # Find ALL vid_* and gt_* subfolders
            # 找到所有 vid_* 和 gt_* 子文件夹
            try:
                all_subdirs = [d for d in os.listdir(subj_path)
                               if os.path.isdir(os.path.join(subj_path, d))]
            except OSError:
                continue

            vid_dirs = {d: os.path.join(subj_path, d)
                        for d in all_subdirs if d.startswith("vid_1m_")}
            gt_dirs = {d: os.path.join(subj_path, d)
                       for d in all_subdirs if d.startswith("gt_1m_")}

            if not vid_dirs:
                print(f"[DailyLoader WARNING] No vid_* folder in {subj_path}, skipping subject.")
                continue

            # Match vid/gt pairs by suffix (vid_1m_XXX ↔ gt_1m_XXX)
            # 通过后缀匹配 vid/gt 对
            matched = 0
            for vid_name, vid_path in sorted(vid_dirs.items()):
                # Extract suffix: "vid_1m_130lux_Stationary_45deg" → "1m_130lux_Stationary_45deg"
                # The suffix is everything after "vid_"
                suffix = vid_name[len("vid_"):]  # e.g. "1m_130lux_Stationary_45deg_useiPhone"
                gt_name = "gt_" + suffix          # e.g. "gt_1m_130lux_Stationary_45deg_useiPhone"

                if gt_name in gt_dirs:
                    gt_path = gt_dirs[gt_name]
                    session_id = suffix  # unique session identifier
                    data_dirs.append({
                        "index": f"subject_{subj_id}_{session_id}",
                        "path": subj_path,
                        "subject": subj_id,
                        "vid_folder": vid_path,
                        "gt_folder": gt_path,
                    })
                    matched += 1
                else:
                    print(f"[DailyLoader WARNING] No matching gt folder for {vid_name} "
                          f"in {subj_path} (expected {gt_name}), skipping session.")

            if matched == 0:
                print(f"[DailyLoader WARNING] No matched vid/gt pairs in {subj_path}, skipping subject.")
            else:
                print(f"  Subject {subj_name}: {matched} session(s) found")

        if not data_dirs:
            raise ValueError(f"No valid vid/gt session pairs found in {data_path}!")

        print(f"  Total: {len(data_dirs)} sessions across {len(subject_folders)} subject folders")
        return data_dirs

    # =================================================================
    #  STEP 2: Split Data by Subject (按受试者划分数据)
    # =================================================================
    def split_raw_data(self, data_dirs, begin, end):
        """Splits data directories by subject to avoid data leakage between splits.

        The split is performed on the subject level, NOT the frame level,
        following rPPG-Toolbox best practices.

        Args:
            data_dirs (List[dict]): Output of get_raw_data().
            begin (float): Fractional start index (0.0–1.0).
            end (float): Fractional end index (0.0–1.0).

        Returns:
            List[dict]: Subset of data_dirs for this split.
        """
        if begin == 0 and end == 1:
            return data_dirs

        # Group by subject and sort
        # 按受试者分组并排序
        data_by_subject = {}
        for d in data_dirs:
            subj = d["subject"]
            if subj not in data_by_subject:
                data_by_subject[subj] = []
            data_by_subject[subj].append(d)

        subject_list = sorted(data_by_subject.keys())
        num_subjects = len(subject_list)

        start_idx = int(begin * num_subjects)
        end_idx = int(end * num_subjects)

        selected = []
        for i in range(start_idx, end_idx):
            subj = subject_list[i]
            selected.extend(data_by_subject[subj])

        return selected

    # =================================================================
    #  STEP 3: Per-Subject Preprocessing Subprocess
    #          (单受试者预处理子流程 — 核心同步逻辑在此)
    # =================================================================
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocesses one subject's data: trim → align → format → save.

        This is invoked by BaseLoader.multi_process_manager() for parallel execution.

        ┌──────────────────────────────────────────────────────────────┐
        │                SYNCHRONIZATION PIPELINE                      │
        │                                                              │
        │  Raw Frames        Raw CSV                                   │
        │  (~4000 PNGs)      (pulse_data.csv)                         │
        │       │                 │                                     │
        │       ▼                 ▼                                     │
        │  ┌─────────────────────────────────┐                         │
        │  │  STEP A: Trimming (修剪)          │                         │
        │  │  Remove first/last 100 frames    │                         │
        │  │  去除首尾各100帧不稳定数据           │                         │
        │  └─────────────┬───────────────────┘                         │
        │                ▼                                              │
        │  ┌─────────────────────────────────┐                         │
        │  │  STEP B: Soft Alignment (软对齐)   │                         │
        │  │  pandas.merge_asof(nearest)      │                         │
        │  │  tolerance = 50ms                │                         │
        │  │  用最近邻匹配，容差50ms               │                         │
        │  └─────────────┬───────────────────┘                         │
        │                ▼                                              │
        │  ┌─────────────────────────────────┐                         │
        │  │  STEP C: Formatting (格式化)       │                         │
        │  │  Resize 128×128, Z-normalize     │                         │
        │  │  Chunk + Save .npy               │                         │
        │  │  缩放、标准化、分块、保存               │                         │
        │  └─────────────────────────────────┘                         │
        └──────────────────────────────────────────────────────────────┘
        """
        data_dir = data_dirs[i]
        saved_filename = data_dir["index"]
        subj_path = data_dir["path"]

        # ------------------------------------------------------------------
        # Load raw data (加载原始数据)
        # Use pre-matched vid/gt folders from get_raw_data()
        # 使用 get_raw_data() 中已匹配的 vid/gt 文件夹
        # ------------------------------------------------------------------
        vid_folder = data_dir["vid_folder"]
        gt_folder = data_dir["gt_folder"]

        # Read frame file list and parse timestamps from filenames
        # 读取帧文件列表，从文件名解析时间戳
        frame_files = sorted(glob.glob(os.path.join(vid_folder, "*.png")))
        if len(frame_files) == 0:
            print(f"[CustomLoader] No PNG frames found in {vid_folder}, skipping.")
            file_list_dict[i] = []
            return

        # Read ground truth CSV
        # 读取真值CSV
        csv_path = os.path.join(gt_folder, "pulse_data.csv")
        gt_df = pd.read_csv(csv_path)

        # ==================================================================
        # STEP A: TRIMMING — Remove Unstable Start/End Frames
        # 步骤A: 修剪 — 去除录制开始和结束时的不稳定帧
        # ==================================================================
        #
        # WHY (为什么):
        #   Recording hardware (cameras, sensors) often produces noisy or
        #   invalid data in the first and last few seconds due to:
        #     - Auto-exposure / auto-focus settling (自动曝光/对焦调整)
        #     - Sensor warm-up (传感器预热)
        #     - Subject movement into position (受试者就位移动)
        #   Trimming 100 frames from each end (~3.3 seconds at 30fps)
        #   eliminates these transient artifacts.
        #
        # HOW (怎么做):
        #   Simply slice the sorted frame list. The CSV will be
        #   automatically trimmed during alignment (Step B) because
        #   only frames that survive trimming participate in merge_asof.
        #
        # ENGLISH:
        #   We discard the first 100 and last 100 frames from the sorted
        #   frame list. This removes the unstable transient periods that
        #   commonly occur at the start and end of physiological recordings.
        #   At a typical 30fps capture rate, 100 frames ≈ 3.3 seconds — a
        #   conservative but effective buffer for hardware settling.
        #
        # 中文:
        #   我们从排序后的帧列表中丢弃前100帧和后100帧。这样可以去除生理信号
        #   录制开始和结束时常见的不稳定过渡期。在典型的30fps采集速率下，
        #   100帧 ≈ 3.3秒 — 这是一个保守但有效的硬件稳定缓冲区。

        total_raw_frames = len(frame_files)
        if total_raw_frames <= (self.TRIM_HEAD + self.TRIM_TAIL):
            print(f"[CustomLoader] Subject {saved_filename}: only {total_raw_frames} frames, "
                  f"too few to trim {self.TRIM_HEAD}+{self.TRIM_TAIL}. Skipping.")
            file_list_dict[i] = []
            return

        frame_files_trimmed = frame_files[self.TRIM_HEAD : total_raw_frames - self.TRIM_TAIL]
        print(f"  [Step A] Trimming: {total_raw_frames} → {len(frame_files_trimmed)} frames "
              f"(removed first {self.TRIM_HEAD} + last {self.TRIM_TAIL})")

        # ==================================================================
        # STEP B: SOFT ALIGNMENT — Nearest Neighbor via merge_asof
        # 步骤B: 软对齐 — 通过merge_asof进行最近邻匹配
        # ==================================================================
        #
        # WHY (为什么):
        #   The video camera and the pulse oximeter / BVP sensor run on
        #   independent clocks with different sampling rates. A frame
        #   captured at t=100ms may have NO exact counterpart in the CSV.
        #   The closest CSV sample might be at t=98ms or t=103ms.
        #   Exact matching would discard almost ALL frames.
        #
        # HOW (怎么做):
        #   pandas.merge_asof performs an efficient O(N log N) merge that,
        #   for each frame timestamp, finds the nearest CSV timestamp.
        #   The `tolerance` parameter (50ms) sets an upper bound: if the
        #   nearest CSV row is still >50ms away, that frame is dropped.
        #
        # ENGLISH:
        #   For every surviving video frame timestamp, we find the CLOSEST
        #   ground-truth timestamp using pandas.merge_asof with
        #   direction='nearest'. This is far more robust than exact matching
        #   because real-world sensors rarely produce perfectly aligned
        #   timestamps. A tolerance of 50ms ensures we don't pair a frame
        #   with a signal sample that is temporally too distant (which would
        #   introduce label noise). Frames without a match within the
        #   tolerance window are dropped.
        #
        # 中文:
        #   对于每一个保留的视频帧时间戳，我们使用 pandas.merge_asof 并设置
        #   direction='nearest' 来查找最接近的真值时间戳。这比精确匹配更加稳健，
        #   因为现实世界的传感器很少产生完美对齐的时间戳。50ms的容差确保我们
        #   不会将一帧与时间上相距过远的信号样本配对（这会引入标签噪声）。
        #   在容差窗口内没有匹配的帧将被丢弃。

        # Build a DataFrame from trimmed frame timestamps
        frame_timestamps = [int(os.path.splitext(os.path.basename(f))[0]) for f in frame_files_trimmed]
        frames_df = pd.DataFrame({
            "frame_ts": frame_timestamps,
            "frame_path": frame_files_trimmed,
        }).sort_values("frame_ts").reset_index(drop=True)

        # Prepare GT DataFrame (ensure sorted by timestamp)
        gt_sorted = gt_df[["PC_Timestamp_ms", "Signal_Value"]].copy()
        gt_sorted = gt_sorted.sort_values("PC_Timestamp_ms").reset_index(drop=True)
        gt_sorted["PC_Timestamp_ms"] = gt_sorted["PC_Timestamp_ms"].astype(np.int64)
        gt_sorted["Signal_Value"] = gt_sorted["Signal_Value"].astype(np.float64)

        # Core alignment: merge_asof with nearest direction
        # 核心对齐: 使用最近邻方向的merge_asof
        frames_df["frame_ts"] = frames_df["frame_ts"].astype(np.int64)

        aligned_df = pd.merge_asof(
            frames_df,
            gt_sorted,
            left_on="frame_ts",
            right_on="PC_Timestamp_ms",
            direction="nearest",
            tolerance=self.ALIGNMENT_TOLERANCE_MS,
        )

        # Drop rows where no match was found (NaN in Signal_Value after merge)
        # 删除未找到匹配的行（合并后Signal_Value为NaN）
        pre_drop_count = len(aligned_df)
        aligned_df = aligned_df.dropna(subset=["Signal_Value"]).reset_index(drop=True)
        post_drop_count = len(aligned_df)
        dropped = pre_drop_count - post_drop_count

        print(f"  [Step B] Alignment: {pre_drop_count} frames → {post_drop_count} matched "
              f"(dropped {dropped} beyond {self.ALIGNMENT_TOLERANCE_MS}ms tolerance)")

        if post_drop_count == 0:
            print(f"[CustomLoader] Subject {saved_filename}: No aligned pairs. Skipping.")
            file_list_dict[i] = []
            return

        # ==================================================================
        # STEP C: FORMATTING — Read, Resize, Normalize, Chunk, Save
        # 步骤C: 格式化 — 读取、缩放、标准化、分块、保存
        # ==================================================================
        aligned_frame_paths = aligned_df["frame_path"].tolist()
        aligned_bvp = aligned_df["Signal_Value"].values.astype(np.float64)

        # C.1: Read video frames from aligned paths (读取对齐后的视频帧)
        frames = self.read_video(aligned_frame_paths)

        # C.2: Read/extract BVP wave from aligned data (提取对齐后的BVP信号)
        bvps = aligned_bvp.copy()

        # C.3: Resample BVP to exactly match frame count (if slight mismatch)
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        # C.4: Apply standard toolbox preprocessing (face crop, resize, normalize, chunk)
        # This calls BaseLoader.preprocess() which handles:
        #   - Face detection & cropping (if enabled in config)
        #   - Resize to config W×H (default 128×128)
        #   - Data type transforms (DiffNormalized, Standardized, Raw)
        #   - Label type transforms
        #   - Chunking into CHUNK_LENGTH clips
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        # C.5: Save preprocessed chunks as .npy files
        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename
        )

        file_list_dict[i] = input_name_list

    # =================================================================
    #  Required Static Methods: read_video / read_wave
    # =================================================================
    @staticmethod
    def read_video(video_file):
        """Reads video frames from a list of image file paths.

        Unlike other loaders that read from a single .avi file, this loader
        reads individual PNG frames whose filenames are timestamps.

        Args:
            video_file: Either a list of PNG file paths (from preprocess_dataset_subprocess)
                        or a single directory path (for retroactive use).

        Returns:
            np.ndarray: Frames array of shape (T, H, W, 3) in RGB order.
        """
        # If given a list of paths (our primary usage)
        if isinstance(video_file, list):
            frame_paths = video_file
        # If given a directory path (fallback / retroactive)
        elif os.path.isdir(video_file):
            frame_paths = sorted(glob.glob(os.path.join(video_file, "*.png")))
        else:
            raise ValueError(f"read_video: expected list of paths or directory, got {type(video_file)}")

        if not frame_paths:
            raise ValueError("read_video: no frames to read!")

        frames = []
        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is None:
                print(f"[WARNING] Could not read frame: {fp}")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads the BVP (pulse) signal from a CSV file.

        Args:
            bvp_file (str): Path to pulse_data.csv.

        Returns:
            np.ndarray: 1-D array of signal values.
        """
        df = pd.read_csv(bvp_file)
        return df["Signal_Value"].values.astype(np.float64)

    # =================================================================
    #  Utility Helpers (辅助工具方法)
    # =================================================================
    @staticmethod
    def _find_subfolder(parent_dir, prefix):
        """Find the first subfolder whose name starts with the given prefix.

        NOTE: This is a legacy helper. The main pipeline now uses get_raw_data()
        which matches ALL vid/gt pairs by suffix. This method is kept for
        backward compatibility and utility use only.

        Args:
            parent_dir (str): Path to search in.
            prefix (str): Prefix to match (e.g., "vid_" or "gt_").

        Returns:
            str or None: Full path to the matched subfolder, or None.
        """
        try:
            candidates = [
                os.path.join(parent_dir, d)
                for d in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith(prefix)
            ]
        except OSError:
            return None

        if not candidates:
            return None

        if len(candidates) > 1:
            print(f"[CustomLoader WARNING] Multiple {prefix}* folders in {parent_dir}. Using: {candidates[0]}")

        return candidates[0]


# =====================================================================
#  EXAMPLE YAML CONFIG TEMPLATE
# =====================================================================
#
# Save as: configs/train_configs/CUSTOM_CUSTOM_TSCAN_BASIC.yaml
#
# -------------------------------------------------------------------
# BASE: ['']
# TOOLBOX_MODE: "train_and_test"
# TRAIN:
#   BATCH_SIZE: 4
#   EPOCHS: 30
#   LR: 9e-3
#   MODEL_FILE_NAME: CustomRPPG_CustomRPPG_TSCAN
#   DATA:
#     FS: 30
#     DATASET: CustomRPPG
#     DO_PREPROCESS: True
#     DATA_PATH: "data/MyDataset"
#     CACHED_PATH: "PreprocessedData/CustomRPPG"
#     BEGIN: 0.0
#     END: 0.8
#     DATA_FORMAT: NDCHW
#     PREPROCESS:
#       DATA_TYPE: ['DiffNormalized', 'Standardized']
#       LABEL_TYPE: DiffNormalized
#       DO_CHUNK: True
#       CHUNK_LENGTH: 180
#       CROP_FACE:
#         DO_CROP_FACE: True
#         BACKEND: 'HC'
#         USE_LARGE_FACE_BOX: True
#         LARGE_BOX_COEF: 1.5
#         DETECTION:
#           DO_DYNAMIC_DETECTION: False
#           DYNAMIC_DETECTION_FREQUENCY: 30
#           USE_MEDIAN_FACE_BOX: False
#       RESIZE:
#         W: 128
#         H: 128
#
# VALID:
#   DATA:
#     FS: 30
#     DATASET: CustomRPPG
#     DO_PREPROCESS: True
#     DATA_PATH: "data/MyDataset"
#     CACHED_PATH: "PreprocessedData/CustomRPPG"
#     BEGIN: 0.8
#     END: 1.0
#     DATA_FORMAT: NDCHW
#     PREPROCESS:
#       DATA_TYPE: ['DiffNormalized', 'Standardized']
#       LABEL_TYPE: DiffNormalized
#       DO_CHUNK: True
#       CHUNK_LENGTH: 180
#       CROP_FACE:
#         DO_CROP_FACE: True
#         BACKEND: 'HC'
#         USE_LARGE_FACE_BOX: True
#         LARGE_BOX_COEF: 1.5
#         DETECTION:
#           DO_DYNAMIC_DETECTION: False
#           DYNAMIC_DETECTION_FREQUENCY: 30
#           USE_MEDIAN_FACE_BOX: False
#       RESIZE:
#         W: 128
#         H: 128
#
# TEST:
#   USE_LAST_EPOCH: True
#   METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']
#   DATA:
#     FS: 30
#     DATASET: CustomRPPG
#     DO_PREPROCESS: True
#     DATA_PATH: "data/MyDataset"
#     CACHED_PATH: "PreprocessedData/CustomRPPG"
#     BEGIN: 0.0
#     END: 1.0
#     DATA_FORMAT: NDCHW
#     PREPROCESS:
#       DATA_TYPE: ['DiffNormalized', 'Standardized']
#       LABEL_TYPE: DiffNormalized
#       DO_CHUNK: True
#       CHUNK_LENGTH: 180
#       CROP_FACE:
#         DO_CROP_FACE: True
#         BACKEND: 'HC'
#         USE_LARGE_FACE_BOX: True
#         LARGE_BOX_COEF: 1.5
#         DETECTION:
#           DO_DYNAMIC_DETECTION: False
#           DYNAMIC_DETECTION_FREQUENCY: 30
#           USE_MEDIAN_FACE_BOX: False
#       RESIZE:
#         W: 128
#         H: 128
#
# MODEL:
#   NAME: Tscan
#   TSCAN:
#     FRAME_DEPTH: 10
#
# INFERENCE:
#   BATCH_SIZE: 4
#   EVALUATION_METHOD: FFT
#   EVALUATION_WINDOW:
#     USE_SMALLER_WINDOW: False
#     WINDOW_SIZE: 10
#
# DEVICE: cuda:0
# NUM_OF_GPU_TRAIN: 1
# LOG:
#   PATH: runs/exp
# -------------------------------------------------------------------