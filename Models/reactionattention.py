from torch.optim.adam import Adam
from Training.losses import *
from Training.evaluation import accuracy, precision_recall
from Training.control import TrainingControl, EarlyStopping
from Modules.linear import LinearClassifier
import torch.nn as nn
from framework.model import Model
from tqdm import tqdm
from utils.plot_curves import precision_recall as pr
from utils.plot_curves import plot_pr_curve
from Modules.reactionattention import ReactionAttentionStack, SelfAttentionStack, AlternateStack, ParallelStack
from Layers.reactionattention import LinearExpansion, ReduceParamLinearExpansion, ConvExpansion, LinearConvExpansion, \
    ShuffleConvExpansion


class ReactionModel_(Model):
    def __init__(
            self, save_path, log_path, dataloader, d_reactant, d_bottleneck, d_classifier,
            d_output, threshold=None, n_layers=6, n_head=8, dropout=0.1,
            stack='ReactionAttention', expansion_layer='LinearExpansion', optimizer=None):

        super().__init__(save_path, log_path)

        # --------------------------- dataloader --------------------------- #
        self.dataloader = dataloader
        feature1_dim, feature2_dim = self.dataloader.get_feature_dim()
        # ^-------------------------- dataloader --------------------------^ #

        # ----------------------------- model ------------------------------ #
        stack_dict = {
            'ReactionAttention': ReactionAttentionStack,
            'SelfAttention': SelfAttentionStack,
            'Alternate': AlternateStack,
            'Parallel': ParallelStack,
        }
        expansion_dict = {
            'LinearExpansion': LinearExpansion,
            'ReduceParamLinearExpansion': ReduceParamLinearExpansion,
            'ConvExpansion': ConvExpansion,
            'LinearConvExpansion': LinearConvExpansion,
            'ShuffleConvExpansion': ShuffleConvExpansion,

        }
        self.model = stack_dict[stack](expansion_dict[expansion_layer], n_layers, n_head, feature1_dim, feature2_dim,
                                       d_reactant, d_bottleneck,
                                       dropout)

        # ^---------------------------- model -----------------------------^ #

        # --------------------------- classifier --------------------------- #
        # self.classifier = LinearClassifier(feature1_dim, d_classifier, d_output)
        self.classifier = nn.Linear(feature1_dim, d_output, bias=False)
        nn.init.xavier_normal_(self.classifier.weight)
        # ^-------------------------- classifier --------------------------^ #

        # If GPU available, move the graph to GPU(s)
        if self.check_cuda():
            # device_ids = list(range(torch.cuda.device_count()))
            # self.model = nn.DataParallel(self.model, device_ids)
            # self.fc1 = nn.DataParallel(self.fc1, device_ids)
            # self.fc2 = nn.DataParallel(self.fc2, device_ids)
            self.model.cuda()
            self.classifier.cuda()

        self.parameters = list(self.model.parameters()) + list(self.classifier.parameters())

        self.d_output = d_output
        self.threshold = threshold

        if optimizer == None:
            self.optimizer = Adam(self.parameters, lr=0.001, betas=(0.9, 0.999))

        # --------------------------- training control --------------------------- #
        self.controller = TrainingControl(max_step=100000, evaluate_every_nstep=100, print_every_nstep=10)
        self.early_stopping = EarlyStopping(patience=50)
        # --------------------------- training control --------------------------- #


    def train_epoch(self, device, smothing):
        ''' Epoch operation in training phase'''

        # check cuda:
        if self.check_cuda():
            pass
        else:
            print('CUDA not enabled, use CPU instead')
            device = 'cpu'

        if self.train_logger == None:
            self.set_logger()

        self.model.train()
        self.classifier.train()

        total_loss = 0
        batch_counter = 0

        for batch in tqdm(
                self.dataloader.train_dataloader(), mininterval=1,
                desc='  - (Training)   ', leave=False):  # training_data should be a iterable

            # prepare data
            feature_1, feature_2, y = map(lambda x: x.to(device), batch)
            # forward
            self.optimizer.zero_grad()
            logits = self.model(feature_1, feature_2)
            logits = self.classifier(logits[0])

            if logits.shape[-1] == 1:
                pred = logits.sigmoid()
                loss = mse_loss(pred, y)

            else:
                pred = logits
                loss = cross_entropy_loss(pred, y, smoothing=smothing)

            acc = accuracy(pred, y, threshold=self.threshold)
            precision, recall, precision_avg, recall_avg = precision_recall(pred, y, self.d_output,
                                                                            threshold=self.threshold)

            loss.backward()

            # update parameters
            self.optimizer.step()

            # note keeping
            total_loss += loss.item()
            batch_counter += 1

            state_dict = self.controller(batch_counter)

            if state_dict['step_to_print']:
                self.train_logger.info(
                    '[TRAINING] - loss: { %3.4f },    acc: { %1.4f },    pre: { %1.4f },    rec: { %1.4f }' % (
                        loss, acc, precision[1], recall[1]))

            if state_dict['step_to_evaluate']:
                stop = self.val_epoch(device, state_dict['step'])
                state_dict['step_to_stop'] = stop
                if stop:
                    break

            if self.controller.current_step == self.controller.max_step:
                state_dict['step_to_stop'] = True
                break

        return state_dict

    def val_epoch(self, device, step):
        ''' Epoch operation in evaluation phase '''

        self.model.eval()
        self.classifier.eval()

        total_loss = 0
        total_acc = 0
        total_pre = 0
        total_rec = 0
        batch_counter = 0
        pred_list = []
        real_list = []

        with torch.no_grad():

            for batch in tqdm(
                    self.dataloader.val_dataloader(), mininterval=5,
                    desc='  - (Evaluation)   ', leave=False):  # training_data should be a iterable

                # prepare data
                feature_1, feature_2, y = map(lambda x: x.to(device), batch)

                logits = self.model(feature_1, feature_2)
                logits = self.classifier(logits[0])

                if logits.shape[-1] == 1:
                    pred = logits.sigmoid()
                    loss = mse_loss(pred, y)

                else:
                    pred = logits
                    loss = cross_entropy_loss(pred, y, smoothing=False)

                acc = accuracy(pred, y, threshold=self.threshold)
                precision, recall, _, _ = precision_recall(pred, y, self.d_output, threshold=self.threshold)

                # note keeping
                total_loss += loss.item()
                total_acc += acc.item()
                total_pre += precision[1].item()
                total_rec += recall[1].item()

                pred_list += pred.tolist()
                real_list += y.tolist()

                batch_counter += 1

            area, precisions, recalls, thresholds = pr(pred_list, real_list)
            plot_pr_curve(recalls, precisions, auc=area)

            loss_avg = total_loss / batch_counter
            acc_avg = total_acc / batch_counter
            pre_avg = total_pre / batch_counter
            rec_avg = total_rec / batch_counter

            self.train_logger.info(
                '[EVALUATION] - loss: { %3.4f },    acc: { %1.4f },    pre: { %1.4f },    rec: { %1.4f }' % (
                    loss_avg, acc_avg, pre_avg, rec_avg))

            state_dict = self.early_stopping(loss_avg)

            if state_dict['save']:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': step}
                self.save_model(checkpoint, self.save_path + '-loss-' + str(loss_avg))

            return state_dict['break']

    def train(self, epoch, device, smoothing, save_mode):
        assert save_mode in ['all', 'best']
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')

            self.controller.set_epoch(epoch_i)
            state_dict = self.train_epoch(device, smoothing)

            if state_dict['step_to_stop']:
                break

        print('[INFO]: Finish Training, ends with %d epoch(s) and %d batches, in total %d training steps.' % (
            state_dict['epoch'] - 1, state_dict['batch'], state_dict['step']))

    def predict(self, data_loader, device):

        pred_list = []
        real_list = []

        self.model.eval()
        self.classifier.eval()

        with torch.no_grad():
            for batch in tqdm(
                    data_loader,
                    desc='  - (Testing)   ', leave=False):
                # prepare data
                feature_1, feature_2, y = map(lambda x: x.to(device), batch)

                logits = self.model(feature_1, feature_2)
                logits = self.classifier(logits[0])

                if logits.shape[-1] == 1:
                    pred = logits.sigmoid()

                else:
                    pred = logits

                pred_list += pred.tolist()
                real_list += y.tolist()

        return pred_list, real_list
