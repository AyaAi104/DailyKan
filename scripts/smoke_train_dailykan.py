"""One-epoch DailyKan smoke training on cached preprocessed data."""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import get_config
from dataset import data_loader
from neural_methods.trainer.DailyKanTrainer import DailyKanTrainer


def _loader_cls(dataset_name):
    if dataset_name == "PURE":
        return data_loader.PURELoader.PURELoader
    if dataset_name == "MMPD":
        return data_loader.MMPDLoader.MMPDLoader
    if dataset_name == "Daily":
        return data_loader.DailyLoader.DailyLoader
    if dataset_name == "Aya":
        return data_loader.AyaLoader.AyaLoader
    raise ValueError(f"Unsupported smoke dataset: {dataset_name}")


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _assert_cached(config):
    for split_name, split in (
        ("TRAIN", config.TRAIN.DATA),
        ("VALID", config.VALID.DATA),
        ("TEST", config.TEST.DATA),
    ):
        if split.DO_PREPROCESS:
            raise ValueError(
                f"{split_name}.DATA.DO_PREPROCESS is True. "
                "Smoke training must use cached data only."
            )


def _build_loader(name, config_data, batch_size, shuffle):
    dataset_cls = _loader_cls(config_data.DATASET)
    dataset = dataset_cls(
        name=name,
        data_path=config_data.DATA_PATH,
        config_data=config_data,
        device=None,
    )
    generator = torch.Generator()
    generator.manual_seed(100)
    return DataLoader(
        dataset=dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _parameter_counts(model):
    module = model.module if hasattr(model, "module") else model
    new_count = sum(p.numel() for p in module.parameters())
    layernorm_count = sum(
        p.numel()
        for name, p in module.named_parameters()
        if ".input_norm." in name
    )
    return new_count - layernorm_count, new_count, layernorm_count


def _count_zero_crossings_per_second(x, fs):
    x = x - np.mean(x)
    signs = np.sign(x)
    signs[signs == 0] = 1
    crossings = np.sum(signs[1:] != signs[:-1])
    seconds = len(x) / float(fs)
    return crossings / 2.0 / max(seconds, 1e-6)


def _run_validation_batch(trainer, valid_loader, save_path):
    trainer.model.eval()
    with torch.no_grad():
        batch = next(iter(valid_loader))
        frames, labels, pose = trainer._unpack_batch(batch, trainer.device)
        pred = trainer.model(frames, pose)
        if torch.isnan(pred).any():
            raise FloatingPointError("NaN detected in validation prediction.")

    pred_cpu = pred.detach().cpu()
    label_cpu = labels.detach().cpu()
    pred0 = pred_cpu[0].numpy()
    label0 = label_cpu[0].numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(pred0, label="pred")
    plt.plot(label0, label="label", alpha=0.8)
    plt.legend()
    plt.title("DailyKan smoke pred vs label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

    pred_std = float(pred_cpu.std(unbiased=False))
    label_std = float(label_cpu.std(unbiased=False))
    ratio = pred_std / max(label_std, 1e-6)
    distinct = int(torch.unique(torch.round(pred_cpu.flatten() * 1000) / 1000).numel())
    osc_per_sec = _count_zero_crossings_per_second(pred0, trainer.config.VALID.DATA.FS)
    return {
        "pred_std": pred_std,
        "label_std": label_std,
        "pred_std_label_std_ratio": ratio,
        "distinct_pred_values_rounded_1e-3": distinct,
        "oscillations_per_second_pred0": osc_per_sec,
        "pred_min": float(pred_cpu.min()),
        "pred_max": float(pred_cpu.max()),
        "label_min": float(label_cpu.min()),
        "label_max": float(label_cpu.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="./configs/train_configs/PURE_MMPD_DailyKan.yaml",
    )
    parser.add_argument("--max_train_batches", type=int, default=50)
    args = parser.parse_args()

    config = get_config(args)
    config.defrost()
    config.TRAIN.EPOCHS = 1
    config.freeze()
    _assert_cached(config)

    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)

    print("Building cached dataloaders...")
    train_loader = _build_loader(
        "train", config.TRAIN.DATA, config.TRAIN.BATCH_SIZE, shuffle=True
    )
    valid_loader = _build_loader(
        "valid", config.VALID.DATA, config.TRAIN.BATCH_SIZE, shuffle=False
    )

    data_loader_dict = {"train": train_loader, "valid": valid_loader, "test": None}
    trainer = DailyKanTrainer(config, data_loader_dict)
    old_count, new_count, delta_count = _parameter_counts(trainer.model)
    print(f"Parameter count old≈{old_count}, new={new_count}, delta={delta_count}")

    first_batch = next(iter(train_loader))
    diagnose_stats = trainer.diagnose_batch(first_batch)
    print("Initial diagnose:")
    print(diagnose_stats)
    if diagnose_stats["kan_basis_alive_frac"] < 0.3:
        print("WARNING: kan_basis_alive_frac < 0.3. Grid range may still be wrong.")

    trainer.model.train()
    var_checkpoints = []
    last_parts = None
    print(f"Running smoke training for at most {args.max_train_batches} batches...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= args.max_train_batches:
            break
        frames, labels, pose = trainer._unpack_batch(batch, trainer.device)
        pred = trainer.model(frames, pose)
        if torch.isnan(pred).any():
            raise FloatingPointError(f"NaN detected in prediction at batch {batch_idx}.")

        labels_norm = trainer._normalize_signal(labels)
        pred_norm = trainer._normalize_signal(pred)
        loss, parts = trainer.loss_model(pred_norm, labels_norm)
        if torch.isnan(loss):
            raise FloatingPointError(f"NaN detected in loss at batch {batch_idx}.")

        trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
        trainer.optimizer.step()
        trainer.scheduler.step()

        last_parts = {k: float(v) for k, v in parts.items()}
        if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
            raw_pred_std = float(pred.detach().std(unbiased=False).cpu())
            raw_label_std = float(labels.detach().std(unbiased=False).cpu())
            print(
                f"batch={batch_idx + 1} "
                f"pred_std={raw_pred_std:.6f} label_std={raw_label_std:.6f} "
                f"pred_min={float(pred.min()):.6f} pred_max={float(pred.max()):.6f} "
                f"loss_parts={last_parts}"
            )
            var_checkpoints.append(last_parts["var"])

    plot_path = os.path.join("runs", "dailykan", "smoke_pred_vs_label.png")
    final_stats = _run_validation_batch(trainer, valid_loader, plot_path)
    print("Final validation batch stats:")
    print(final_stats)
    print(f"Saved plot: {plot_path}")

    checks = {
        "pred_std_label_std_ratio_gt_0.3": final_stats["pred_std_label_std_ratio"] > 0.3,
        "distinct_pred_values_gt_20": final_stats["distinct_pred_values_rounded_1e-3"] > 20,
        "kan_basis_alive_frac_gt_0.5": diagnose_stats["kan_basis_alive_frac"] > 0.5,
        "var_monotonic_nonincreasing": all(
            b <= a + 1e-4 for a, b in zip(var_checkpoints, var_checkpoints[1:])
        ) if len(var_checkpoints) > 1 else True,
        "pred0_oscillations_per_sec_ge_1": final_stats["oscillations_per_second_pred0"] >= 1.0,
    }
    print("Smoke checks:")
    print(checks)

    if all(checks.values()):
        print("PASS: DailyKan smoke criteria passed.")
    else:
        failed = [name for name, ok in checks.items() if not ok]
        print("FAIL: DailyKan smoke criteria failed:")
        for name in failed:
            print(f"  - {name}")
        if not checks["kan_basis_alive_frac_gt_0.5"]:
            print("Likely adjustment: KAN grid/input normalization still needs attention.")
        if not checks["pred_std_label_std_ratio_gt_0.3"] or not checks["distinct_pred_values_gt_20"]:
            print("Likely adjustment: anti-collapse loss weights or output head scale need review.")
        if not checks["var_monotonic_nonincreasing"]:
            print("Likely adjustment: variance term is unstable; inspect early learning rate/clip.")
        if not checks["pred0_oscillations_per_sec_ge_1"]:
            print("Likely adjustment: spectral/SNR pressure may still be too weak.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
