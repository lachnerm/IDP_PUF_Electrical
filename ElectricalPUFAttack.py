import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import Accuracy

from attack_models.Aseeri2018 import Aseeri2018Model, weights_init_Aseeri2018
from attack_models.Custom import CustomModel, weights_init_Custom
from attack_models.Mursi2020 import Mursi2020Model, weights_init_Mursi2020
from attack_models.Wisiol2022 import Wisiol2022Model, weights_init_Wisiol2022

matplotlib.use('Agg')


class ElectricalPUFModule(LightningModule):
    def __init__(self, hparams, args):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.do_log = args.do_log
        self.log_interval = 5

        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

        self.train_accs = []
        self.val_accs = []
        self.test_accs = 0

        device = 'cuda' if args.accelerator == 'gpu' else 'cpu'
        self.train_counts_pred = torch.zeros(2, device=device)
        self.val_counts_pred = torch.zeros(2, device=device)
        self.test_counts_pred = torch.zeros(2, device=device)

        self.criterion = nn.BCEWithLogitsLoss()

        if args.a == 'Mursi':
            self.model = Mursi2020Model(args.bits, args.stages)
            self.model.apply(weights_init_Mursi2020)
        elif args.a == 'Aseeri':
            self.model = Aseeri2018Model(args.bits, args.stages)
            self.model.apply(weights_init_Aseeri2018)
        elif args.a == 'Wisiol':
            self.model = Wisiol2022Model(args.bits)
            self.model.apply(weights_init_Wisiol2022)
        elif args.a == 'Custom':
            self.model = CustomModel(args.bits, args.stages)
            self.model.apply(weights_init_Custom)
        else:
            raise RuntimeError(f'Unknown architecture {args.a}')

    def training_step(self, batch, batch_idx):
        loss, preds, real_response_adj = self._run_step(batch)
        self.train_acc(preds, real_response_adj)
        self.log('train_acc', self.train_acc, prog_bar=True)

        if self.do_log and self.current_epoch % self.log_interval == 0:
            self._add_class_count(preds, self.train_counts_pred)

        return loss

    def training_epoch_end(self, outputs):
        self.train_accs.append(self.train_acc.compute().item())
        self.log('Train Accuracy', self.train_acc)

        if self.do_log:
            outputs = [output["loss"] for output in outputs]
            self._log_at_epoch_end(outputs, stage='Train')
            self.train_counts_pred = torch.zeros(2, device=self.device)

    def validation_step(self, batch, batch_idx):
        loss, preds, real_response_adj = self._run_step(batch)
        self.val_acc(preds, real_response_adj)
        self.log('val_acc', self.val_acc, prog_bar=True)

        if self.do_log and self.current_epoch % self.log_interval == 0:
            self._add_class_count(preds, self.val_counts_pred)

        return loss

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc.compute().item()
        self.val_accs.append(val_acc)
        self.log('Val Accuracy', self.val_acc, on_epoch=True)

        if self.do_log:
            self._log_at_epoch_end(outputs, stage='Val')
            self.val_counts_pred = torch.zeros(2, device=self.device)

    def test_step(self, batch, batch_idx):
        loss, preds, real_response_adj = self._run_step(batch)
        self.test_acc(preds, real_response_adj)

        if self.do_log and self.current_epoch % self.log_interval == 0:
            self._add_class_count(preds, self.test_counts_pred)

        return loss

    def test_epoch_end(self, outputs):
        self.test_accs = self.test_acc.compute().item()
        self.log('Test Accuracy', self.test_acc, on_epoch=True)

        if self.do_log:
            self._log_at_epoch_end(outputs, stage='Test')
            self.val_counts_pred = torch.zeros(2, device=self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.hparams.lr,
            (0.9, 0.999)
        )
        return optimizer

    def backward(self, trainer, loss, optimizer_idx):
        super().backward(trainer, loss, optimizer_idx)
        if self.do_log and self.global_step % self.log_interval == 0:
            self.logger.experiment.add_figure(
                'Gradient',
                self._plot_grad_flow(self.model.named_parameters()),
                self.current_epoch
            )

    def _run_step(self, batch):
        challenge, real_response = batch
        real_response = real_response.squeeze()
        real_response_adj = torch.where(real_response == 1, real_response, 0)
        gen_response = self.model(challenge).squeeze()
        loss = self.criterion(gen_response, real_response_adj)

        gen_response = gen_response.sigmoid()
        preds = gen_response.round()

        return loss, preds, real_response_adj

    def _log_at_epoch_end(self, outputs, stage):
        loss = torch.stack(outputs).mean()
        self.logger.experiment.add_scalars("Loss", {f'{stage} Loss': loss},
                                           self.current_epoch)
        if self.current_epoch % self.log_interval == 0:
            self.logger.experiment.add_figure(
                'Train Class Preds',
                self._get_class_barplot(self.train_counts_pred),
                self.current_epoch
            )
            plt.close()

    def _add_class_count(self, data, self_counts):
        vals, counts = torch.unique(data, return_counts=True)
        for idx, cnts in zip(vals, counts):
            self_counts[idx.int().item()] += cnts

    def _get_class_barplot(self, counts):
        counts = counts.tolist()
        df = pd.DataFrame({
            '0': counts[0],
            '1': counts[1]
        }, index=[0])
        g = sns.barplot(df)
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        return g.figure

    def _plot_grad_flow(self, named_parameters):
        '''
        Plots the gradients flowing through different layers in the net during training. Assumes that a figure was
        initiated beforehand.
        '''
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.set_title('Model Gradient Flow')
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())

        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1,
               color="c")
        ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1,
               color="b")
        ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        ax.set_xticks(range(0, len(ave_grads), 1))
        ax.set_xticklabels(layers, rotation="vertical")
        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(bottom=-0.001, top=0.2)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Average gradient")
        ax.grid(True)
        ax.legend([Line2D([0], [0], color="c", lw=4),
                   Line2D([0], [0], color="b", lw=4),
                   Line2D([0], [0], color="k", lw=4)],
                  ['max-gradient', 'mean-gradient', 'zero-gradient'])
        return fig
