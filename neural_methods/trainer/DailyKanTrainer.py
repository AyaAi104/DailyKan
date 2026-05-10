"""Trainer for DailyKan."""

import os

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.FrequencyDomainLoss import FrequencyDomainLoss
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.DailyKan import DailyKan
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class DailyKanTrainer(BaseTrainer):

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
        frame_num = data_cfg.PREPROCESS.CHUNK_LENGTH
        self.chunk_len = frame_num
        self.freq_loss_weight = float(getattr(config.MODEL.DAILYKAN, "FREQ_LOSS_WEIGHT", 0.2))
        self.loss_model = Neg_Pearson()
        self.freq_loss_model = FrequencyDomainLoss(fs=data_cfg.FS).to(self.device)
        self.model = DailyKan(
            in_channels=in_channels,
            frames=frame_num,
            width=config.MODEL.DAILYKAN.WIDTH,
            grid_size=config.MODEL.DAILYKAN.GRID_SIZE,
            pose_hidden=config.MODEL.DAILYKAN.POSE_HIDDEN,
            dropout=config.MODEL.DROP_RATE,
        ).to(self.device)
        self.model = torch.nn.DataParallel(
            self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
        )

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.TRAIN.LR,
                weight_decay=config.MODEL.DAILYKAN.WEIGHT_DECAY,
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
            raise ValueError("DailyKan trainer initialized in incorrect toolbox mode!")

    @staticmethod
    def _unpack_batch(batch, device):
        if len(batch) < 5:
            raise ValueError(
                "DailyKan requires pose angles. Preprocess with "
                "PREPROCESS.CROP_FACE.BACKEND: 'INSIGHT' so *_pose*.npy files are created."
            )
        frames = batch[0].to(torch.float32).to(device)
        labels = batch[1].to(torch.float32).to(device)
        pose = batch[4].to(torch.float32).to(device)
        # BaseLoader NDCHW returns [B,T,C,H,W]; DailyKan expects [B,C,T,H,W].
        if frames.ndim == 5 and frames.shape[2] in (3, 6, 9):
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        if pose.ndim == 3 and pose.shape[1] != 3 and pose.shape[2] == 3:
            pose = pose.permute(0, 2, 1).contiguous()
        return frames, labels, pose

    @staticmethod
    def _normalize_signal(x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

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
                frames, labels, pose = self._unpack_batch(batch, self.device)
                pred_ppg = self.model(frames, pose)
                labels = self._normalize_signal(labels)
                pred_ppg = self._normalize_signal(pred_ppg)
                time_loss = self.loss_model(pred_ppg, labels)
                freq_loss = self.freq_loss_model(pred_ppg, labels)
                loss = time_loss + self.freq_loss_weight * freq_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
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
                    time=time_loss.item(),
                    freq=freq_loss.item(),
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
                frames, labels, pose = self._unpack_batch(valid_batch, self.device)
                pred_ppg = self.model(frames, pose)
                labels = self._normalize_signal(labels)
                pred_ppg = self._normalize_signal(pred_ppg)
                time_loss = self.loss_model(pred_ppg, labels)
                freq_loss = self.freq_loss_model(pred_ppg, labels)
                loss = time_loss + self.freq_loss_weight * freq_loss
                valid_loss.append(loss.item())
                vbar.set_postfix(
                    loss=loss.item(),
                    time=time_loss.item(),
                    freq=freq_loss.item(),
                )
        return np.mean(np.asarray(valid_loss))

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("")
        print("===Testing===")
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
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
            self.model.load_state_dict(torch.load(model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                frames, label, pose = self._unpack_batch(test_batch, self.config.DEVICE)
                pred_ppg_test = self.model(frames, pose)

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
