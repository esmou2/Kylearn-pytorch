from Training.optimizers import AdamOptimizer
from Training.losses import *
from Training.evaluation import accuracy, precision_racall
from framework.model import Model
import torch.nn as nn

import pytorch_lightning as pl


class ReactionModelLightning(pl.LightningModule):
    def __init__(self, dataloader, stack, d_reactant, d_bottleneck, d_classifier, d_output, threshold=None,
                 n_layers=6, n_head=8, dropout=0.1):
        super().__init__()

        self.dataloader = dataloader
        feature1_dim, feature2_dim = self.dataloader.get_feature_dim()
        self.model = stack(n_layers, n_head, feature1_dim, feature2_dim, d_reactant, d_bottleneck,
                                            dropout)
        self.fc1 = nn.Linear(feature1_dim, d_classifier)
        self.fc2 = nn.Linear(d_classifier, d_output)

        self.threshold = threshold

    def forward(self, feature_1, feature_2):
        output = self.model(feature_1, feature_2)
        output = self.fc1(output[0])
        output = nn.functional.relu(output)
        output = self.fc2(output)
        return output

    def training_step(self, batch, batch_nb):
        feature_1, feature_2, y = batch
        logits = self.forward(feature_1, feature_2)
        pred = logits.sigmoid()


        if y.shape[-1] == 1:
            loss = mse_loss(pred, y)

        else:
            loss = cross_entropy_loss(pred, y, smoothing=True)

        acc = accuracy(pred, y, threshold=self.threshold)
        _, _, precision_avg, recall_avg = precision_racall(pred, y, threshold=self.threshold)
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc,
                            'train_precision': precision_avg,
                            'train_recall': recall_avg}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        feature_1, feature_2, y = batch
        logits = self.forward(feature_1, feature_2)
        pred = logits.sigmoid()

        if y.shape[-1] == 1:
            loss = mse_loss(pred, y)

        else:
            loss = cross_entropy_loss(pred, y, smoothing=False)

        acc = accuracy(pred, y, threshold=self.threshold)
        _, _, precision_avg, recall_avg = precision_racall(pred, y, threshold=self.threshold)

        return {'val_loss': loss, 'val_acc': acc,
                'val_precision': precision_avg,
                'val_recall': recall_avg}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc,
                            'val_precision': avg_precision,
                            'val_recall': avg_recall}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return AdamOptimizer(self.parameters(), 1000, 0.01)

    @pl.data_loader
    def train_dataloader(self):
        return self.dataloader.train_dataloader()

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloader.val_dataloader()

    @pl.data_loader
    def test_dataloader(self):
        return self.dataloader.test_dataloader()


class ReactionModel(Model):
    def __init__(self, save_path, log_path, dataloader, stack, d_reactant, d_bottleneck, d_classifier, d_output, threshold=None,
                 n_layers=6, n_head=8, dropout=0.1):

        super().__init__(save_path)

        self.dataloader = dataloader
        feature1_dim, feature2_dim = self.dataloader.get_feature_dim()

        self.model = stack(n_layers, n_head, feature1_dim, feature2_dim, d_reactant, d_bottleneck,
                           dropout)

        self.fc1 = nn.Linear(feature1_dim, d_classifier)
        self.fc2 = nn.Linear(d_classifier, d_output)

        self.threshold = threshold

        self._logger_train = logger(log_path + 'train')
        self._logger_eval = logger(log_path + 'test')

    def train_epoch(self, training_data, device, smoothing):
        ''' Epoch operation in training phase'''

        self.model.train()

        total_loss = 0
        n_sample_total = 0
        n_sample_correct = 0

        for batch in tqdm(
                training_data, mininterval=1,
                desc='  - (Training)   ', leave=False):  # training_data should be a iterable

            # TODO: batch iterable
            # prepare data
            index, position, target = map(lambda x: x.to(device), batch)

            # forward
            self.optimizer.zero_grad()
            self.logits = self.model(index, position)

            # backward
            if self.is_regression:
                # activation
                self.pred = nn.Sigmoid(self.logits)

                loss = self.loss_regression(self.logits, target)
                n_correct = self.performance_regression(self.logits, target, threshold=self.threshold)

            else:
                loss = self.loss_cross_entropy(self.logits, target, smoothing)
                n_correct = self.performance_multi(self.logits, target)

            loss.backward()

            # update parameters
            self.optimizer.step()

            # note keeping
            total_loss += loss.item()

            non_pad_mask = target.ne(0)
            n_sample = non_pad_mask.sum().item()
            n_sample_total += n_sample
            n_sample_correct += n_correct

        loss_per_word = total_loss / n_sample_total
        accuracy = n_sample_correct / n_sample_total
        return loss_per_word, accuracy

    def eval_epoch(self, validation_data, device):
        ''' Epoch operation in evaluation phase '''

        self.model.eval()

        total_loss = 0
        n_sample_total = 0
        n_sample_correct = 0

        with torch.no_grad():

            # prepare data
            index, position, target = map(lambda x: x.to(device), validation_data)

            # forward
            logits = self.model(index, position)

            if self.is_regression:
                loss = self.loss_regression(logits, target)
                n_correct = self.performance_regression(logits, target, threshold=self.threshold)

            else:
                loss = self.loss_cross_entropy(logits, target)
                n_correct = self.performance_multi(logits, target)

            # note keeping
            total_loss += loss.item()

            # note keeping
            total_loss += loss.item()

            non_pad_mask = target.ne(0)
            n_sample = non_pad_mask.sum().item()
            n_sample_total += n_sample
            n_sample_correct += n_correct

        loss_per_word = total_loss / n_sample_total
        accuracy = n_sample_correct / n_sample_total
        return loss_per_word, accuracy

    def train(self, training_data, validation_data, epoch, device, smoothing, save_mode):
        assert save_mode in ['all', 'best']
        valid_losses = []
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')

            start = time.time()
            train_loss, train_accu = self.train_epoch(training_data, device, smoothing=smoothing)
            self._logger_train.info('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
                                    'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
                elapse=(time.time() - start) / 60))

            start = time.time()
            valid_loss, valid_accu = self.eval_epoch(validation_data, device)
            self._logger_eval.info('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
                                   'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
                elapse=(time.time() - start) / 60))

            valid_losses += [valid_loss]

            checkpoint = {
                'models': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch_i}

            if save_mode == 'all':
                self.save_model(checkpoint, self.save_path + '_loss_{loss:3.3f}.chkpt'.format(loss=valid_loss))

            if save_mode == 'best':
                if valid_loss >= max(valid_loss):
                    self.save_model(checkpoint, self.save_path + '_loss_{loss:3.3f}.chkpt'.format(loss=valid_loss))
                    print('    - [Info] The checkpoint file has been updated.')
