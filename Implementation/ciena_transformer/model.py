from framework.model import Model
from Modules.linear import LinearClassifier
from Modules.transformer import *
from Layers.transformer import *
from torch.optim.adamw import AdamW
from Training.losses import *
from Training.evaluation import accuracy, precision_recall, Evaluator
from Training.control import TrainingControl, EarlyStopping
from tqdm import tqdm


def get_attn_mask(padding):
    '''
        Arguments:
            padding {Tensor, float32 [batch, t*max_length, 1]} -- padding index, 1 means is padded
        Returns:
            non_pad_mask {Tensor, [batch, t*max_length, 1]} -- 0 means is padded
            slf_attn_mask {Tensor, [batch, t*max_length, t*max_length]} -- True means is padded
    '''
    padding = padding.squeeze(-1)
    slf_attn_mask = get_attn_key_pad_mask(seq_k=padding, seq_q=padding,
                                          padding_idx=1)  # [batch, t*max_length, t*max_length]
    non_pad_mask = torch.cuda.FloatTensor(1) - padding.unsqueeze(-1)  # [batch, t*max_length, 1]
    non_pad_mask = non_pad_mask.float()

    return non_pad_mask, slf_attn_mask


class CienaTransformerModel(Model):
    def __init__(
            self, save_path, log_path, d_features, d_meta, max_length, d_classifier, n_classes, threshold=None,
            optimizer=None, **kwargs):
        '''**kwargs: n_layers, n_head, dropout, use_bottleneck, d_bottleneck'''
        '''
            Arguments:
                save_path -- model file path
                log_path -- log file path
                d_features -- how many PMs
                d_meta -- how many facility types
                max_length -- input sequence length
                d_classifier -- classifier hidden unit
                n_classes -- output dim
                threshold -- if not None, n_classes should be 1 (regression).
        '''

        super().__init__(save_path, log_path)
        self.d_output = n_classes
        self.threshold = threshold
        self.max_length = max_length

        # ----------------------------- Model ------------------------------ #

        self.model = Encoder(TimeFacilityEncoding, d_features=d_features, max_seq_length=max_length,
                                       d_meta=d_meta, **kwargs)

        # --------------------------- Classifier --------------------------- #
        self.classifier = LinearClassifier(d_features * max_length, d_classifier, n_classes)

        # ------------------------------ CUDA ------------------------------ #
        # If GPU available, move the graph to GPU(s)
        self.CUDA_AVAILABLE = self.check_cuda()
        if self.CUDA_AVAILABLE:
            device_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids)
            self.classifier = nn.DataParallel(self.classifier, device_ids)
            self.model.to('cuda')
            self.classifier.to('cuda')
            assert (next(self.model.parameters()).is_cuda)
            assert (next(self.classifier.parameters()).is_cuda)
            pass

        else:
            print('CUDA not found or not enabled, use CPU instead')

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

            input_feature_sequence, padding, time_facility, y = map(lambda x: x.to(device), batch)

            batch_size = len(y)

            non_pad_mask, slf_attn_mask = get_attn_mask(padding)

            # forward
            self.optimizer.zero_grad()
            logits, attn = self.model(input_feature_sequence, time_facility, non_pad_mask, slf_attn_mask)
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

                input_feature_sequence, padding, time_facility, y = map(lambda x: x.to(device), batch)

                batch_size = len(y)

                non_pad_mask, slf_attn_mask = get_attn_mask(padding)

                # get logits
                logits, attn = self.model(input_feature_sequence, time_facility, non_pad_mask, slf_attn_mask)
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
            #     if plot:
            #         pred_list += pred.tolist()
            #         real_list += y.tolist()
            #
            # if plot:
            #     area, precisions, recalls, thresholds = pr(pred_list, real_list)
            #     plot_pr_curve(recalls, precisions, auc=area)

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

                input_feature_sequence, padding, time_facility, y = map(lambda x: x.to(device), batch)

                non_pad_mask, slf_attn_mask = get_attn_mask(padding)

                # get logits
                logits, attn = self.model(input_feature_sequence, time_facility, non_pad_mask, slf_attn_mask)
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

