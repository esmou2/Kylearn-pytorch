from framework.model import Model
from Modules.transformer import Transformer, TransformerCls
from Optimizer.adam import AdamOptimizer
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from utils.loggings import logger
from abc import abstractmethod


class TransformerModel(Model):
    def __init__(self, save_path, log_path):
        super().__init__(save_path=save_path)

        self.model = None
        self.optimizer = None
        self._logger_train = logger(log_path + 'train')
        self._logger_eval = logger(log_path + 'test')

    def loss_cross_entropy(self, logits, real, smoothing=False):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        real = real.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = logits.size(1)

            one_hot = torch.zeros_like(logits).scatter(1, real.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(logits, dim=1)

            non_pad_mask = real.ne(0)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(logits, real, ignore_index=0, reduction='sum')

        return loss

    def performance_multi(self, logits, real):
        ''' Apply label smoothing if needed '''

        logits = logits.max(1)[1]
        real = real.contiguous().view(-1)
        non_pad_mask = real.ne(0)
        n_correct = logits.eq(real)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return n_correct

    def loss_regression(self, logits, real):
        loss = F.mse_loss(logits, real)

        return loss

    def performance_regression(self, logits, real, threshold):
        ''' Apply label smoothing if needed '''

        logits = logits.ge(threshold)
        real = real.contiguous().view(-1)
        non_pad_mask = real.ne(0)
        n_correct = logits.eq(real)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return n_correct


    @abstractmethod
    def train_epoch(self, training_data, device, smoothing):
        ''' Epoch operation in training phase'''
        pass
        # self.model.train()
        #
        # total_loss = 0
        # n_word_total = 0
        # n_word_correct = 0
        #
        # for batch in tqdm(
        #         training_data, mininterval=2,
        #         desc='  - (Training)   ', leave=False):  # process bar
        #     # prepare data
        #     src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        #     real = tgt_seq[:, 1:]
        #
        #     # forward
        #     self.optimizer.zero_grad()
        #     pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
        #
        #     # backward
        #     loss, n_correct = self.cal_performance_multi(pred, real, smoothing=smoothing)
        #     loss.backward()
        #
        #     # update parameters
        #     self.optimizer.step()
        #
        #     # note keeping
        #     total_loss += loss.item()
        #
        #     non_pad_mask = real.ne(0)
        #     n_word = non_pad_mask.sum().item()
        #     n_word_total += n_word
        #     n_word_correct += n_correct
        #
        # loss_per_word = total_loss / n_word_total
        # accuracy = n_word_correct / n_word_total
        # return loss_per_word, accuracy

    @abstractmethod
    def eval_epoch(self, validation_data, device):
        ''' Epoch operation in evaluation phase '''
        pass
        # self.model.eval()
        #
        # total_loss = 0
        # n_word_total = 0
        # n_word_correct = 0
        #
        # with torch.no_grad():
        #     for batch in tqdm(
        #             validation_data, mininterval=2,
        #             desc='  - (Validation) ', leave=False):
        #         # prepare data
        #         src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        #         gold = tgt_seq[:, 1:]
        #
        #         # forward
        #         pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
        #         loss, n_correct = self.cal_performance_multi(pred, gold, smoothing=False)
        #
        #         # note keeping
        #         total_loss += loss.item()
        #
        #         non_pad_mask = gold.ne(0)
        #         n_word = non_pad_mask.sum().item()
        #         n_word_total += n_word
        #         n_word_correct += n_correct
        #
        # loss_per_word = total_loss / n_word_total
        # accuracy = n_word_correct / n_word_total
        # return loss_per_word, accuracy

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


class TransformerClsModel(TransformerModel):
    def __init__(self, save_path, log_path, embedding, len_max_seq, output_dim,
                 d_word_vec=512, d_model=512, d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
                 warmup_step=100, lr=0.01, is_regression=False, threshold=0):

        super().__init__(save_path, log_path)
        self.model = TransformerCls(embedding=embedding, len_max_seq=len_max_seq,
                                    d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                                    n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                    output_num=output_dim, dropout=dropout)

        self.optimizer = AdamOptimizer(self.model.parameters(), warmup_step, lr_max=lr)
        self.is_regression = is_regression
        self.threshold = threshold

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
            index, position, target= map(lambda x: x.to(device), batch)

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
