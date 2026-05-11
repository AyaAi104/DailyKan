"""Run DailyKan forward diagnostics on one cached batch."""

import argparse
import os
import random
import sys

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
    raise ValueError(f"Unsupported diagnosis dataset: {dataset_name}")


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _assert_cached(config):
    for split in (config.TRAIN.DATA, config.VALID.DATA, config.TEST.DATA):
        if split.DO_PREPROCESS:
            raise ValueError("diagnose_dailykan.py requires DO_PREPROCESS: False for all splits.")


def _build_train_loader(config):
    dataset_cls = _loader_cls(config.TRAIN.DATA.DATASET)
    dataset = dataset_cls(
        name="train",
        data_path=config.TRAIN.DATA.DATA_PATH,
        config_data=config.TRAIN.DATA,
        device=config.DEVICE,
    )
    generator = torch.Generator()
    generator.manual_seed(100)
    return DataLoader(
        dataset=dataset,
        num_workers=0,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
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
    return {
        "old_approx_without_kan_layernorm": new_count - layernorm_count,
        "new": new_count,
        "delta": layernorm_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="./configs/train_configs/PURE_MMPD_DailyKan.yaml",
    )
    args = parser.parse_args()
    config = get_config(args)
    _assert_cached(config)

    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)

    train_loader = _build_train_loader(config)
    trainer = DailyKanTrainer(config, {"train": train_loader, "valid": None, "test": None})
    batch = next(iter(train_loader))
    frames, labels, pose = trainer._unpack_batch(batch, trainer.device)

    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    stats = model.diagnose(frames, pose)
    print("DailyKan diagnosis:")
    print(stats)
    print("Parameter counts:")
    print(_parameter_counts(trainer.model))
    if stats["kan_basis_alive_frac"] < 0.3:
        print("WARNING: KAN basis alive fraction < 0.3. Grid range or normalization may still be wrong.")


if __name__ == "__main__":
    main()
