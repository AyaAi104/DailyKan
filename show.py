# plot_pure_ppg_from_cache.py
import os, glob, numpy as np
import matplotlib.pyplot as plt

# === 配置你的路径（把这行改成你 YAML 里的 CACHED_PATH） ===
CACHED_PATH = r"D:/rppg_cache"   # e.g. "D:/rppg_cache" 或 "/mnt/d/rppg_cache"
FS = 30.0                        # 与 YAML 里 FS 一致（PURE/SCAMPS 常用 30）

# 找一条 PURE 的 label
pp_dir = os.path.join(CACHED_PATH, "PURE_SizeW128_SizeH128_ClipLength160_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse")
label_files = sorted(glob.glob(os.path.join(pp_dir, "**", "*label*.npy"), recursive=True))
if not label_files:
    raise FileNotFoundError(f"没有找到 PURE 的 label npy 文件，请确认已完成预处理，检查目录：{pp_dir}")

label_path = label_files[0]
ppg = np.load(label_path).astype(float).squeeze()   # [T] 或 [1,T]
t = np.arange(len(ppg)) / FS

print("Loaded label:", label_path, "length:", len(ppg), "samples @", FS, "Hz")

plt.figure()
plt.plot(t, ppg)
plt.xlabel("Time (s)")
plt.ylabel("PPG (a.u.)")
plt.title("PURE PPG (time domain)")
plt.tight_layout()
plt.show()
