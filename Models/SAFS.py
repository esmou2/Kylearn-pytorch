from framework.model import Model
from Modules.linear import LinearClassifier
import torch
import torch.nn as nn
import numpy as np
from torch.optim.adamw import AdamW
from Training.losses import *
from Training.evaluation import accuracy, precision_recall, Evaluator
from Training.control import TrainingControl, EarlyStopping
from Layers.expansions import feature_shuffle_index
from tqdm import tqdm
from utils.plot_curves import precision_recall as pr
from utils.plot_curves import plot_pr_curve


def parse_data(batch, device):
    # get data from dataloader
    try:
        feature_1, feature_2, y = map(lambda x: x.to(device), batch)
    except:
        feature_1, y = map(lambda x: x.to(device), batch)
        feature_2 = None

    return feature_1, feature_2, y


class ScaledDotProduction(nn.Module):
    '''Scaled Dot Production'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):
        '''
            Arguments:
                query {Tensor, shape: [batch, d_k, d_out]} -- query
                key {Tensor, shape: [batch, d_k, n_candidate]} -- key
                value {Tensor, shape: [batch, d_v, n_candidate]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, n_depth, n_vchannel * d_features] -- output
                attn {Tensor, shape [n_head * batch, n_depth, n_depth] -- reaction attention
        '''
        attn = torch.bmm(query.transpose(2, 1), key)  # [batch, d_out, n_candidate]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.bmm(attn, value.transpose(2, 1))  # [batch, d_out, d_v]

        return output, attn


class Bottleneck(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        residual = features
        features = self.layer_norm(features)

        features = self.w_2(F.relu(self.w_1(features)))
        features = self.dropout(features)
        features += residual

        return features


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_features, d_out, kernel, stride, d_k, d_v, n_replica, shuffled_index, dropout=0.1):
        super().__init__()
        self.d_features = d_features
        self.d_out = d_out
        self.stride = np.ceil(np.divide(d_features, d_out)).astype(int)
        self.n_replica = n_replica
        self.shuffled_index = shuffled_index

        self.query = nn.Conv1d(1, d_k, self.stride, self.stride, bias=False)
        self.key = nn.Conv1d(1, d_k, kernel, stride, bias=False)
        self.value = nn.Conv1d(1, d_v, kernel, stride, bias=False)
        self.conv = nn.Conv1d(d_v, 1, 1, 1, bias=False)

        self.query.initialize_param(nn.init.xavier_normal_)
        self.key.initialize_param(nn.init.xavier_normal_)
        self.value.initialize_param(nn.init.xavier_normal_)
        nn.init.xavier_normal(self.conv.weight)


        self.attention = ScaledDotProduction(temperature=1)

        self.layer_norm = nn.LayerNorm(d_features)
        self.dropout = nn.Dropout(dropout)

        ### Use Bottleneck? ###
        self.bottleneck = Bottleneck(d_out, d_out)


    def forward(self, features):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_features]} -- features

            Returns:
                output {Tensor, shape [batch, d_features]} -- output
                attn {Tensor, shape [n_head * batch, d_features, d_features]} -- self attention
        '''
        d_features, d_out, n_replica, shuffled_index \
            = self.d_features, self.d_out, self.n_replica, self.shuffled_index

        residual = features

        query = self.layer_norm(features)

        # d_features_ceil = d_out * stride
        #
        # features = F.pad(features, (0, d_features_ceil - d_features), value=0)  # shape: [batch, d_features_ceil]

        shuffled_features = features[:, self.index]  # shape: [batch, n_replica * d_features]

        query = self.query(query)  # shape: [batch, d_k, d_out]
        key = self.key(
            shuffled_features)  # shape: [batch, d_k, n_candidate], n_candidate = (n_replica * d_features) / stride
        value = self.value(shuffled_features)  # shape: [batch, d_v, n_candidate]

        output, attn = self.attention(query, key, value)  # shape: [batch, d_out, d_v], [batch, d_out, n_candidate]
        output = output.transpose(2, 1).contiguous()  # shape: [batch, d_v, d_out]
        output = self.conv(output).view(-1, d_out)  # shape: [batch, d_out]
        output = self.dropout(output)

        if d_features == d_out:
            output += residual

        ### Use Bottleneck? ###
        output = self.bottleneck(output)

        return output, attn


class SelfAttentionFeatureSelection(nn.Module):
    def __init__(self, d_features, d_out_list):
        super().__init__()
        self.index = feature_shuffle_index(d_features, 8)
        self.layers = nn.ModuleList([
            SelfAttentionLayer(d_features, d_out, kernel=3, stride=3, d_k=32, d_v=32, n_replica=8,
                               shuffled_index=self.index) for d_out in d_out_list])

    def forward(self, features, feature_2=None):
        self_attn_list = []

        for layer in self.layers:
            features, self_attn = layer(
                features)
            self_attn_list += [self_attn]

        return features, self_attn_list


class SAFSModel(Model):
    def __init__(
            self, save_path, log_path, d_features, d_out_list, d_classifier, d_output, threshold=None, optimizer=None,
            **kwargs):
        '''*args: n_layers, n_head, n_channel, n_vchannel, dropout'''
        super().__init__(save_path, log_path)
        self.d_output = d_output
        self.threshold = threshold

        # ----------------------------- Model ------------------------------ #

        self.model = SelfAttentionFeatureSelection(d_features, d_out_list)

        # --------------------------- Classifier --------------------------- #

        self.classifier = LinearClassifier(d_features, d_classifier, d_output)

        # ------------------------------ CUDA ------------------------------ #
        self.data_parallel()

        # ---------------------------- Optimizer --------------------------- #
        self.parameters = list(self.model.parameters()) + list(self.classifier.parameters())
        if optimizer == None:
            self.optimizer = AdamW(self.parameters, lr=0.002, betas=(0.9, 0.999), weight_decay=0.001)

        # ------------------------ training control ------------------------ #
        self.controller = TrainingControl(max_step=100000, evaluate_every_nstep=100, print_every_nstep=10)
        self.early_stopping = EarlyStopping(patience=50)

        # --------------------- logging and tensorboard -------------------- #
        self.set_logger()
        self.set_summary_writer()
        # ---------------------------- END INIT ---------------------------- #

    def checkpoint(self, step):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': step}
        return checkpoint

    def train_epoch(self, train_dataloader, eval_dataloader, device, smothing, earlystop):
        ''' Epoch operation in training phase'''

        if device == 'cuda':
            assert self.CUDA_AVAILABLE
        # Set model and classifier training mode
        self.model.train()
        self.classifier.train()

        total_loss = 0
        batch_counter = 0

        # update param per batch
        for batch in tqdm(
                train_dataloader, mininterval=1,
                desc='  - (Training)   ', leave=False):  # training_data should be a iterable

            # get data from dataloader

            feature_1, feature_2, y = parse_data(batch, device)

            batch_size = len(feature_1)

            # forward
            self.optimizer.zero_grad()
            logits, attn = self.model(feature_1, feature_2)
            logits = logits.view(batch_size, -1)
            logits = self.classifier(logits)

            # Judge if it's a regression problem
            if self.d_output == 1:
                pred = logits.sigmoid()
                loss = mse_loss(pred, y)

            else:
                pred = logits
                loss = cross_entropy_loss(pred, y, smoothing=smothing)

            # calculate gradients
            loss.backward()

            # update parameters
            self.optimizer.step()

            # get metrics for logging
            acc = accuracy(pred, y, threshold=self.threshold)
            precision, recall, precision_avg, recall_avg = precision_recall(pred, y, self.d_output,
                                                                            threshold=self.threshold)
            total_loss += loss.item()
            batch_counter += 1

            # training control
            state_dict = self.controller(batch_counter)

            if state_dict['step_to_print']:
                self.train_logger.info(
                    '[TRAINING]   - step: %5d, loss: %3.4f, acc: %1.4f, pre: %1.4f, rec: %1.4f' % (
                        state_dict['step'], loss, acc, precision[1], recall[1]))
                self.summary_writer.add_scalar('loss/train', loss, state_dict['step'])
                self.summary_writer.add_scalar('acc/train', acc, state_dict['step'])
                self.summary_writer.add_scalar('precision/train', precision[1], state_dict['step'])
                self.summary_writer.add_scalar('recall/train', recall[1], state_dict['step'])

            if state_dict['step_to_evaluate']:
                stop = self.val_epoch(eval_dataloader, device, state_dict['step'])
                state_dict['step_to_stop'] = stop

                if earlystop & stop:
                    break

            if self.controller.current_step == self.controller.max_step:
                state_dict['step_to_stop'] = True
                break

        return state_dict

    def val_epoch(self, dataloader, device, step=0, plot=False):
        ''' Epoch operation in evaluation phase '''
        if device == 'cuda':
            assert self.CUDA_AVAILABLE

        # Set model and classifier training mode
        self.model.eval()
        self.classifier.eval()

        # use evaluator to calculate the average performance
        evaluator = Evaluator()

        pred_list = []
        real_list = []

        with torch.no_grad():

            for batch in tqdm(
                    dataloader, mininterval=5,
                    desc='  - (Evaluation)   ', leave=False):  # training_data should be a iterable

                # get data from dataloader
                feature_1, feature_2, y = parse_data(batch, device)

                batch_size = len(feature_1)

                # get logits
                logits, attn = self.model(feature_1, feature_2)
                logits = logits.view(batch_size, -1)
                logits = self.classifier(logits)

                if self.d_output == 1:
                    pred = logits.sigmoid()
                    loss = mse_loss(pred, y)

                else:
                    pred = logits
                    loss = cross_entropy_loss(pred, y, smoothing=False)

                acc = accuracy(pred, y, threshold=self.threshold)
                precision, recall, _, _ = precision_recall(pred, y, self.d_output, threshold=self.threshold)

                # feed the metrics in the evaluator
                evaluator(loss.item(), acc.item(), precision[1].item(), recall[1].item())

                '''append the results to the predict / real list for drawing ROC or PR curve.'''
                if plot:
                    pred_list += pred.tolist()
                    real_list += y.tolist()

            if plot:
                area, precisions, recalls, thresholds = pr(pred_list, real_list)
                plot_pr_curve(recalls, precisions, auc=area)

            # get evaluation results from the evaluator
            loss_avg, acc_avg, pre_avg, rec_avg = evaluator.avg_results()

            self.eval_logger.info(
                '[EVALUATION] - step: %5d, loss: %3.4f, acc: %1.4f, pre: %1.4f, rec: %1.4f' % (
                    step, loss_avg, acc_avg, pre_avg, rec_avg))
            self.summary_writer.add_scalar('loss/eval', loss_avg, step)
            self.summary_writer.add_scalar('acc/eval', acc_avg, step)
            self.summary_writer.add_scalar('precision/eval', pre_avg, step)
            self.summary_writer.add_scalar('recall/eval', rec_avg, step)

            state_dict = self.early_stopping(loss_avg)

            if state_dict['save']:
                checkpoint = self.checkpoint(step)
                self.save_model(checkpoint, self.save_path + '-step-%d_loss-%.5f' % (step, loss_avg))

            return state_dict['break']

    def train(self, max_epoch, train_dataloader, eval_dataloader, device,
              smoothing=False, earlystop=False, save_mode='best'):

        assert save_mode in ['all', 'best']
        # train for n epoch
        for epoch_i in range(max_epoch):
            print('[ Epoch', epoch_i, ']')
            # set current epoch
            self.controller.set_epoch(epoch_i + 1)
            # train for on epoch
            state_dict = self.train_epoch(train_dataloader, eval_dataloader, device, smoothing, earlystop)

            # if state_dict['step_to_stop']:
            #     break

        checkpoint = self.checkpoint(state_dict['step'])

        self.save_model(checkpoint, self.save_path + '-step-%d' % state_dict['step'])

        self.train_logger.info(
            '[INFO]: Finish Training, ends with %d epoch(s) and %d batches, in total %d training steps.' % (
                state_dict['epoch'] - 1, state_dict['batch'], state_dict['step']))

    def get_predictions(self, data_loader, device, max_batches=None, activation=None):

        pred_list = []
        real_list = []

        self.model.eval()
        self.classifier.eval()

        batch_counter = 0

        with torch.no_grad():
            for batch in tqdm(
                    data_loader,
                    desc='  - (Testing)   ', leave=False):

                feature_1, feature_2, y = parse_data(batch, device)

                # get logits
                logits, attn = self.model(feature_1, feature_2)
                logits = logits.view(logits.shape[0], -1)
                logits = self.classifier(logits)

                # Whether to apply activation function
                if activation != None:
                    pred = activation(logits)
                else:
                    pred = logits.softmax(dim=-1)
                pred_list += pred.tolist()
                real_list += y.tolist()

                if max_batches != None:
                    batch_counter += 1
                    if batch_counter >= max_batches:
                        break

        return pred_list, real_list
