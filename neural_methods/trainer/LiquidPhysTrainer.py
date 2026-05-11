"""Trainer for LiquidPhys."""

import os

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.LiquidPhys import LiquidPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class LiquidPhysTrainer(BaseTrainer):
    """Train, validate, and test the LiquidPhys model."""

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        data_cfg = config.TRAIN.DATA if config.TOOLBOX_MODE == "train_and_test" else config.TEST.DATA
        in_channels = 3 * len(data_cfg.PREPROCESS.DATA_TYPE)
        self.chunk_len = data_cfg.PREPROCESS.CHUNK_LENGTH
        self.frame_rate = data_cfg.FS
        self.freq_loss_weight = config.MODEL.LIQUIDPHYS.FREQ_LOSS_WEIGHT
        motion_dim = 3 + config.MODEL.LIQUIDPHYS.FLOW_DIM

        self.model = LiquidPhys(
            in_channels=in_channels,
            embed_dim=config.MODEL.LIQUIDPHYS.EMBED_DIM,
            hidden_dim=config.MODEL.LIQUIDPHYS.HIDDEN_DIM,
            motion_dim=motion_dim,
            motion_embed_dim=config.MODEL.LIQUIDPHYS.MOTION_EMBED_DIM,
            dropout=config.MODEL.DROP_RATE,
            cfc_mode=config.MODEL.LIQUIDPHYS.CFC_MODE,
            mixed_memory=config.MODEL.LIQUIDPHYS.MIXED_MEMORY,
        ).to(self.device)
        self.model = torch.nn.DataParallel(
            self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
        )
        self.criterion_pearson = Neg_Pearson()

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.TRAIN.LR,
                weight_decay=config.MODEL.LIQUIDPHYS.WEIGHT_DECAY,
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches,
            )
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("LiquidPhys trainer initialized in incorrect toolbox mode!")

    @staticmethod
    def _unpack_batch(batch, device):
        if len(batch) < 6:
            raise ValueError(
                "LiquidPhys requires pose and optical-flow features. Preprocess with "
                "PREPROCESS.USE_POSE: True, PREPROCESS.USE_OPTICAL_FLOW: True, "
                "and CROP_FACE.BACKEND: 'INSIGHT'."
            )
        frames = batch[0].to(torch.float32).to(device)
        labels = batch[1].to(torch.float32).to(device)
        pose = batch[4].to(torch.float32).to(device)
        flow = batch[5].to(torch.float32).to(device)
        return frames, labels, pose, flow

    @staticmethod
    def _normalize_signal(x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def _band_psd_distribution(self, x):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        window = torch.hann_window(x.shape[-1], dtype=x.dtype, device=x.device)
        psd = torch.fft.rfft(x * window, dim=-1).abs().pow(2)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(self.frame_rate)).to(x.device)
        mask = (freqs >= 0.7) & (freqs <= 3.0)
        psd = psd[:, mask]
        return psd / psd.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _frequency_kl_loss(self, pred, label):
        pred_psd = self._band_psd_distribution(pred)
        label_psd = self._band_psd_distribution(label)
        kl_pred_label = pred_psd * (torch.log(pred_psd + 1e-8) - torch.log(label_psd + 1e-8))
        kl_label_pred = label_psd * (torch.log(label_psd + 1e-8) - torch.log(pred_psd + 1e-8))
        return torch.mean(0.5 * (kl_pred_label.sum(dim=-1) + kl_label_pred.sum(dim=-1)))

    def _compute_loss(self, pred_ppg, labels):
        labels = self._normalize_signal(labels)
        pred_ppg = self._normalize_signal(pred_ppg)
        pearson = self.criterion_pearson(pred_ppg, labels)
        freq = self._frequency_kl_loss(pred_ppg, labels)
        total = pearson + self.freq_loss_weight * freq
        return total, {"pearson": pearson.detach(), "freq": freq.detach(), "total": total.detach()}

    def diagnose_batch(self, batch):
        frames, labels, pose, flow = self._unpack_batch(batch, self.device)
        model = self.model.module if hasattr(self.model, "module") else self.model
        shapes = model.diagnose(frames, pose, flow)
        pred = self.model(frames, pose, flow)
        shapes["label"] = tuple(labels.shape)
        shapes["pred_runtime"] = tuple(pred.shape)
        return shapes

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print("")
            print(f"====Training Epoch: {epoch}====")
            train_loss = []
            running_loss = 0.0
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                frames, labels, pose, flow = self._unpack_batch(batch, self.device)
                pred_ppg = self.model(frames, pose, flow)
                loss, parts = self._compute_loss(pred_ppg, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                lrs.append(self.scheduler.get_last_lr())

                running_loss += loss.item()
                train_loss.append(loss.item())
                if idx % 100 == 99:
                    print(f"[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0
                tbar.set_postfix(
                    loss=loss.item(),
                    pearson=float(parts["pearson"]),
                    freq=float(parts["freq"]),
                )

            mean_training_losses.append(np.mean(train_loss))
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print("validation loss: ", valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("")
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_batch in vbar:
                vbar.set_description("Validation")
                frames, labels, pose, flow = self._unpack_batch(valid_batch, self.device)
                pred_ppg = self.model(frames, pose, flow)
                loss, parts = self._compute_loss(pred_ppg, labels)
                valid_loss.append(loss.item())
                vbar.set_postfix(
                    loss=loss.item(),
                    pearson=float(parts["pearson"]),
                    freq=float(parts["freq"]),
                )
        return np.mean(np.asarray(valid_loss))

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("")
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            model_path = self.config.INFERENCE.MODEL_PATH
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + "_Epoch" + str(self.max_epoch_num - 1) + ".pth",
                )
                print("Testing uses last epoch as non-pretrained model!")
            else:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + "_Epoch" + str(self.best_epoch) + ".pth",
                )
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
        print(model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                frames, label, pose, flow = self._unpack_batch(test_batch, self.config.DEVICE)
                pred_ppg_test = self.model(frames, pose, flow)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print("")
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + "_Epoch" + str(index) + ".pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print("Saved Model Path: ", model_path)
