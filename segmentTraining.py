import argparse
import datetime
import os
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from util.logconf import logging

from wnet import WNet, SegmentationAugmentation
from segmentDsets import Luna2dSegmentationDataset

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

METRICS_LOSS_NDX = 0
METRICS_TP_NDX = 1
METRICS_FN_NDX = 2
METRICS_FP_NDX = 3
METRICS_SIZE = 4


class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            default=16,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            default=10,
                            type=int,
                            )
        parser.add_argument('--lr',
                            default=0.001,
                            type=float,
                            )
        parser.add_argument('--augmented',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-flip',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-offset',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-scale',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-rotate',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-noise',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--tb-prefix',
                            default='wnet_seg',
                            )
        parser.add_argument('comment',
                            nargs='?',
                            default='dw',
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.augmentation_model = self.initAugmentation()

    def initModel(self):
        model = WNet(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )
        if self.use_cuda:
            log.info(f"Using CUDA with {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initAugmentation(self):
        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                augmentation_model = nn.DataParallel(augmentation_model)
            augmentation_model = augmentation_model.to(self.device)
        return augmentation_model

    def initOptimizer(self):
        return Adam(self.model.parameters(), lr=self.cli_args.lr)

    def initTrainDl(self):
        train_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            shuffle=True,
        )
        return train_dl

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir=f"{log_dir}_trn_{self.cli_args.comment}")
            self.val_writer = SummaryWriter(log_dir=f"{log_dir}_val_{self.cli_args.comment}")

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info(f"Epoch {epoch_ndx}/{self.cli_args.epochs}, "
                     f"{len(train_dl)}/{len(val_dl)} batches of size "
                     f"{self.cli_args.batch_size}*{torch.cuda.device_count() if self.use_cuda else 1}")

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            best_score = max(score, best_score)

            self.saveModel(epoch_ndx, score == best_score)

        self.trn_writer.close()
        self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            batch_iter = enumerateWithEstimate(
                val_dl,
                f"E{epoch_ndx} Validation",
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, classificationThreshold=0.5):
        input_t, label_t, _, _ = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction1_g, prediction2_g = self.model(input_g)

        dice_loss1 = self.diceLoss(prediction1_g, label_g)
        dice_loss2 = self.diceLoss(prediction2_g, label_g)

        loss_g = dice_loss1 + dice_loss2

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction2_g > classificationThreshold).to(torch.float32)

            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (~label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return loss_g.mean()

    def diceLoss(self, prediction_g, label_g, epsilon=1e-6):
        intersection = (prediction_g * label_g).sum(dim=[1, 2, 3])
        union = prediction_g.sum(dim=[1, 2, 3]) + label_g.sum(dim=[1, 2, 3])

        dice = (2. * intersection + epsilon) / (union + epsilon)

        return 1 - dice

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)

        all_pos = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]
        all_pred_pos = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
        metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] / (all_pos if all_pos > 0 else 1)
        metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] / (all_pred_pos if all_pred_pos > 0 else 1)
        metrics_dict['pr/f1_score'] = 2 * (metrics_dict['pr/precision'] * metrics_dict['pr/recall']) / \
                                      (metrics_dict['pr/precision'] + metrics_dict['pr/recall'] if (metrics_dict[
                                                                                                        'pr/precision'] +
                                                                                                    metrics_dict[
                                                                                                        'pr/recall']) > 0 else 1)

        log.info(f"E{epoch_ndx} {mode_str} ---- "
                 f"Loss: {metrics_dict['loss/all']:.4f}, "
                 f"Recall: {metrics_dict['pr/recall']:.4f}, "
                 f"Precision: {metrics_dict['pr/precision']:.4f}, "
                 f"F1: {metrics_dict['pr/f1_score']:.4f}")

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(f'seg/{key}', value, self.totalTrainingSamples_count)
        writer.flush()

        return metrics_dict['pr/f1_score']

    def saveModel(self, epoch_ndx, isBest=False):
        model_path = os.path.join('models', self.cli_args.tb_prefix, f"{self.time_str}_{self.cli_args.comment}.state")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, model_path)
        log.info(f"Saved model to {model_path}")

        if isBest:
            best_path = os.path.join('models', self.cli_args.tb_prefix, f"{self.cli_args.comment}.best.state")
            torch.save(state, best_path)
            log.info(f"Saved best model to {best_path}")


if __name__ == '__main__':
    SegmentationTrainingApp().main()
