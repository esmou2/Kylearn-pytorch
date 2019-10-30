from torch.optim.adam import Adam
from Training.losses import *
from Training.evaluation import accuracy, precision_recall, Evaluator
from Training.control import TrainingControl, EarlyStopping
from Modules.linear import LinearClassifier
from framework.model import Model
from tqdm import tqdm
from utils.plot_curves import precision_recall as pr
from utils.plot_curves import plot_pr_curve
from Modules.attention import *
from Layers.expansions import *

def parse_data(batch, device):
    # get data from dataloader
    try:
        feature_1, feature_2, y = map(lambda x: x.to(device), batch)
    except:
        feature_1, y = map(lambda x: x.to(device), batch)
        feature_2 = None

    return feature_1, feature_2, y


class ShuffleSelfAttentionModel(Model):
    def __init__(
            self, save_path, log_path, n_depth, d_features, d_classifier, d_output, threshold=None,
            stack='ShuffleSelfAttention', expansion_layer='ChannelWiseConvExpansion', mode='1d', optimizer=None, **kwargs):
        '''*args: n_layers, n_head, n_channel, n_vchannel, dropout, use_bottleneck, d_bottleneck'''

        super().__init__(save_path, log_path)
        self.d_output = d_output
        self.threshold = threshold

        # ----------------------------- Model ------------------------------ #
        stack_dict = {
            'ReactionAttention': ReactionAttentionStack,
            'SelfAttention': SelfAttentionStack,
            'Alternate': AlternateStack,
            'Parallel': ParallelStack,
            'ShuffleSelfAttention': ShuffleSelfAttentionStack
        }
        expansion_dict = {
            'LinearExpansion': LinearExpansion,
            'ReduceParamLinearExpansion': ReduceParamLinearExpansion,
            'ConvExpansion': ConvExpansion,
            'LinearConvExpansion': LinearConvExpansion,
            'ShuffleConvExpansion': ShuffleConvExpansion,
            'ChannelWiseConvExpansion': ChannelWiseConvExpansion,
        }

        self.model = stack_dict[stack](expansion_dict[expansion_layer], n_depth=n_depth, d_features=d_features,
                                       mode=mode, **kwargs)

        # --------------------------- Classifier --------------------------- #
        if mode == '1d':
            self.classifier = LinearClassifier(d_features, d_classifier, d_output)
        elif mode == '2d':
            self.classifier = LinearClassifier(n_depth * d_features, d_classifier, d_output)
        else:
            self.classifier = None

        # ------------------------------ CUDA ------------------------------ #
        # If GPU available, move the graph to GPU(s)

        # device_ids = list(range(torch.cuda.device_count()))
        # self.model = nn.DataParallel(self.model, device_ids)
        # self.fc1 = nn.DataParallel(self.fc1, device_ids)
        # self.fc2 = nn.DataParallel(self.fc2, device_ids)]
        self.CUDA_AVAILABLE = self.check_cuda()
        if self.CUDA_AVAILABLE:
            self.model.cuda()
            self.classifier.cuda()
        else:
            print('CUDA not found or not enabled, use CPU instead')

        # ---------------------------- Optimizer --------------------------- #
        self.parameters = list(self.model.parameters()) + list(self.classifier.parameters())
        if optimizer == None:
            self.optimizer = Adam(self.parameters, lr=0.001, betas=(0.9, 0.999))

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
            self.summary_writer.add_scalar('precision/eval', rec_avg, step)


            state_dict = self.early_stopping(loss_avg)

            if state_dict['save']:
                checkpoint = self.checkpoint(step)
                self.save_model(checkpoint, self.save_path + '-step-%d_loss-%.5f'%(step,loss_avg))

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

        self.train_logger.info('[INFO]: Finish Training, ends with %d epoch(s) and %d batches, in total %d training steps.' % (
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


