from framework.model import Model
from Modules.transformer import Transformer
from Optimizer.adam import AdamOptimizer
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from utils.loggings import logger


class TransformerModel(Model):
    def __init__(self, save_path, log_path,
                 n_src_vocab, n_tgt_vocab, len_max_seq,
                 d_word_vec=512, d_model=512, d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
                 warmup_step=100, lr=0.01,
                 tgt_emb_prj_weight_sharing=True,
                 emb_src_tgt_weight_sharing=True):
        super().__init__(save_path=save_path)

        self.model = Transformer(n_src_vocab, n_tgt_vocab, len_max_seq,
                                 d_word_vec, d_model, d_inner,
                                 n_layers, n_head, d_k, d_v, dropout,
                                 tgt_emb_prj_weight_sharing,
                                 emb_src_tgt_weight_sharing)

        self.optimizer = AdamOptimizer(self.model.parameters(), warmup_step, lr_max=lr)

        self.logger_train = logger(log_path + 'train')
        self.logger_eval = logger(log_path + 'test')

    def loss(self, pred, real, smoothing):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        real = real.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, real.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = real.ne(0)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, real, ignore_index=0, reduction='sum')

        return loss

    def cal_performance(self, pred, real, smoothing=False):
        ''' Apply label smoothing if needed '''

        loss = self.loss(pred, real, smoothing)

        pred = pred.max(1)[1]
        real = real.contiguous().view(-1)
        non_pad_mask = real.ne(0)
        n_correct = pred.eq(real)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct

    def train_epoch(self, training_data, device, smoothing):
        ''' Epoch operation in training phase'''

        self.model.train()

        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        for batch in tqdm(
                training_data, mininterval=2,
                desc='  - (Training)   ', leave=False):  # process bar
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            real = tgt_seq[:, 1:]

            # forward
            self.optimizer.zero_grad()
            pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)

            # backward
            loss, n_correct = self.cal_performance(pred, real, smoothing=smoothing)
            loss.backward()

            # update parameters
            self.optimizer.step()

            # note keeping
            total_loss += loss.item()

            non_pad_mask = real.ne(0)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def eval_epoch(self, validation_data, device):
        ''' Epoch operation in evaluation phase '''

        self.model.eval()

        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        with torch.no_grad():
            for batch in tqdm(
                    validation_data, mininterval=2,
                    desc='  - (Validation) ', leave=False):
                # prepare data
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
                gold = tgt_seq[:, 1:]

                # forward
                pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
                loss, n_correct = self.cal_performance(pred, gold, smoothing=False)

                # note keeping
                total_loss += loss.item()

                non_pad_mask = gold.ne(0)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def train(self, training_data, validation_data, epoch, optimizer, device, smoothing, save_mode):
        assert save_mode in ['all', 'best']
        valid_losses = []
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')

            start = time.time()
            train_loss, train_accu = self.train_epoch(training_data, device, smoothing=smoothing)
            self.logger_train.info('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
                  'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
                elapse=(time.time() - start) / 60))

            start = time.time()
            valid_loss, valid_accu = self.eval_epoch(validation_data, device)
            self.logger_eval.info('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
                  'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
                elapse=(time.time() - start) / 60))

            valid_losses += [valid_loss]

            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch_i}

            if save_mode == 'all':
                self.save_model(checkpoint, self.save_path + '_loss_{loss:3.3f}.chkpt'.format(loss=valid_loss))

            if save_mode == 'best':
                if valid_loss >= max(valid_loss):
                    self.save_model(checkpoint, self.save_path + '_loss_{loss:3.3f}.chkpt'.format(loss=valid_loss))
                    print('    - [Info] The checkpoint file has been updated.')

