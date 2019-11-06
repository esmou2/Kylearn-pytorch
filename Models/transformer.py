from framework.model import Model
from Modules.linear import LinearClassifier
from Modules.transformer import *
from Layers.transformer import *
from torch.optim.adamw import AdamW
from Training.losses import *
from Training.evaluation import accuracy, precision_recall, Evaluator
from Training.control import TrainingControl, EarlyStopping
from tqdm import tqdm

def parse_data_enc(input_sequence, embedding):
    '''
    Returns:
        enc_output {Tensor, [batch_size, seq_length, d_v]} --
        non_pad_mask {Tensor, [n_head, seq_length, 1]} --
        slf_attn_mask {Tensor, [batch_size, seq_length, seq_length]} --
    '''

    slf_attn_mask = get_attn_key_pad_mask(seq_k=input_sequence, seq_q=input_sequence,
                                          padding_idx=0)  # [batch_size, seq_length, seq_length]
    non_pad_mask = get_non_pad_mask(input_sequence, padding_idx=0)  # [batch_size, seq_length, 1]

    embedding_sequence = embedding(input_sequence)

    return embedding_sequence, non_pad_mask, slf_attn_mask


def parse_data_dec(input_sequence, target_sequence, embedding):
    non_pad_mask = get_non_pad_mask(target_sequence)

    slf_attn_mask_subseq = get_subsequent_mask(target_sequence)
    slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=target_sequence, seq_q=target_sequence)
    slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

    dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_sequence, seq_q=target_sequence)

    embedding_sequence = embedding(target_sequence)

    return embedding_sequence, non_pad_mask, slf_attn_mask, dec_enc_attn_mask


class TransormerClassifierModel(Model):
    def __init__(
            self, save_path, log_path, d_features, d_meta, max_length, d_classifier, n_classes, threshold=None, embedding=None,
            stack='Encoder', position_encode='SinusoidPositionEncoding', optimizer=None, **kwargs):
        '''**kwargs: n_layers, n_head, dropout, use_bottleneck, d_bottleneck'''

        super().__init__(save_path, log_path)
        self.d_output = n_classes
        self.threshold = threshold
        self.max_length = max_length
        
        # ----------------------------- Model ------------------------------ #
        stack_dict = {
            'Encoder': Encoder,
            'Transformer': Transformer
        }
        encoding_dict = {
            'SinusoidPositionEncoding': SinusoidPositionEncoding,
            'LinearPositionEncoding': LinearPositionEncoding
        }

        self.model = stack_dict[stack](encoding_dict[position_encode], d_features=d_features, max_seq_length=max_length, d_meta=d_meta, **kwargs)
        
        # --------------------------- Embedding  --------------------------- #
        if embedding != None:
            self.word_embedding = nn.Embedding.from_pretrained(embedding)
            self.USE_EMBEDDING = True
        else:
            self.USE_EMBEDDING = False
            
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

            index, position, y = map(lambda x: x.to(device), batch)
            
            batch_size = len(index)

            input_feature_sequence, non_pad_mask, slf_attn_mask = parse_data_enc(index, self.word_embedding)

            # forward
            self.optimizer.zero_grad()
            logits, attn = self.model(input_feature_sequence, position)
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

                index, position, y = map(lambda x: x.to(device), batch)

                batch_size = len(index)

                input_feature_sequence, non_pad_mask, slf_attn_mask = parse_data_enc(index, self.word_embedding)

                # get logits
                logits, attn = self.model(input_feature_sequence, position)
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

                index, position, y = map(lambda x: x.to(device), batch)

                input_feature_sequence, non_pad_mask, slf_attn_mask = parse_data_enc(index, self.word_embedding)

                # get logits
                logits, attn = self.model(input_feature_sequence, position)
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



# class TransformerModel(Model):
#     def __init__(self, save_path, log_path):
#         super().__init__(save_path=save_path, log_path=log_path)
#
#         self.model = None
#         self.optimizer = None
#         self._logger_train = logger(log_path + 'train')
#         self._logger_eval = logger(log_path + 'test')
#
#     def loss_cross_entropy(self, logits, real, smoothing=False):
#         ''' Calculate cross entropy loss, apply label smoothing if needed. '''
#
#         real = real.contiguous().view(-1)
#
#         if smoothing:
#             eps = 0.1
#             n_class = logits.size(1)
#
#             one_hot = torch.zeros_like(logits).scatter(1, real.view(-1, 1), 1)
#             one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#             log_prb = F.log_softmax(logits, dim=1)
#
#             non_pad_mask = real.ne(0)
#             loss = -(one_hot * log_prb).sum(dim=1)
#             loss = loss.masked_select(non_pad_mask).sum()  # average later
#         else:
#             loss = F.cross_entropy(logits, real, ignore_index=0, reduction='sum')
#
#         return loss
#
#     def performance_multi(self, logits, real):
#         ''' Apply label smoothing if needed '''
#
#         logits = logits.max(1)[1]
#         real = real.contiguous().view(-1)
#         non_pad_mask = real.ne(0)
#         n_correct = logits.eq(real)
#         n_correct = n_correct.masked_select(non_pad_mask).sum().item()
#
#         return n_correct
#
#     def loss_regression(self, logits, real):
#         loss = F.mse_loss(logits, real)
#
#         return loss
#
#     def performance_regression(self, logits, real, threshold):
#         ''' Apply label smoothing if needed '''
#
#         logits = logits.ge(threshold)
#         real = real.contiguous().view(-1)
#         non_pad_mask = real.ne(0)
#         n_correct = logits.eq(real)
#         n_correct = n_correct.masked_select(non_pad_mask).sum().item()
#
#         return n_correct
#
#
#     @abstractmethod
#     def train_epoch(self, training_data, device, smoothing):
#         ''' Epoch operation in training phase'''
#         pass
#         # self.model.train()
#         #
#         # total_loss = 0
#         # n_word_total = 0
#         # n_word_correct = 0
#         #
#         # for batch in tqdm(
#         #         training_data, mininterval=2,
#         #         desc='  - (Training)   ', leave=False):  # process bar
#         #     # prepare data
#         #     src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
#         #     real = tgt_seq[:, 1:]
#         #
#         #     # forward
#         #     self.optimizer.zero_grad()
#         #     pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
#         #
#         #     # backward
#         #     loss, n_correct = self.cal_performance_multi(pred, real, smoothing=smoothing)
#         #     loss.backward()
#         #
#         #     # update parameters
#         #     self.optimizer.step()
#         #
#         #     # note keeping
#         #     total_loss += loss.item()
#         #
#         #     non_pad_mask = real.ne(0)
#         #     n_word = non_pad_mask.sum().item()
#         #     n_word_total += n_word
#         #     n_word_correct += n_correct
#         #
#         # loss_per_word = total_loss / n_word_total
#         # accuracy = n_word_correct / n_word_total
#         # return loss_per_word, accuracy
#
#     @abstractmethod
#     def eval_epoch(self, validation_data, device):
#         ''' Epoch operation in evaluation phase '''
#         pass
#         # self.model.eval()
#         #
#         # total_loss = 0
#         # n_word_total = 0
#         # n_word_correct = 0
#         #
#         # with torch.no_grad():
#         #     for batch in tqdm(
#         #             validation_data, mininterval=2,
#         #             desc='  - (Validation) ', leave=False):
#         #         # prepare data
#         #         src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
#         #         gold = tgt_seq[:, 1:]
#         #
#         #         # forward
#         #         pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
#         #         loss, n_correct = self.cal_performance_multi(pred, gold, smoothing=False)
#         #
#         #         # note keeping
#         #         total_loss += loss.item()
#         #
#         #         non_pad_mask = gold.ne(0)
#         #         n_word = non_pad_mask.sum().item()
#         #         n_word_total += n_word
#         #         n_word_correct += n_correct
#         #
#         # loss_per_word = total_loss / n_word_total
#         # accuracy = n_word_correct / n_word_total
#         # return loss_per_word, accuracy
#
#     def train(self, training_data, validation_data, epoch, device, smoothing, save_mode):
#         assert save_mode in ['all', 'best']
#         valid_losses = []
#         for epoch_i in range(epoch):
#             print('[ Epoch', epoch_i, ']')
#
#             start = time.time()
#             train_loss, train_accu = self.train_epoch(training_data, device, smoothing=smoothing)
#             self._logger_train.info('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
#                                     'elapse: {elapse:3.3f} min'.format(
#                 ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
#                 elapse=(time.time() - start) / 60))
#
#             start = time.time()
#             valid_loss, valid_accu = self.eval_epoch(validation_data, device)
#             self._logger_eval.info('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
#                                    'elapse: {elapse:3.3f} min'.format(
#                 ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
#                 elapse=(time.time() - start) / 60))
#
#             valid_losses += [valid_loss]
#
#             checkpoint = {
#                 'models': self.model.state_dict(),
#                 'optimizer_state_dict': self.optimizer.state_dict(),
#                 'epoch': epoch_i}
#
#             if save_mode == 'all':
#                 self.save_model(checkpoint, self.save_path + '_loss_{loss:3.3f}.chkpt'.format(loss=valid_loss))
#
#             if save_mode == 'best':
#                 if valid_loss >= max(valid_loss):
#                     self.save_model(checkpoint, self.save_path + '_loss_{loss:3.3f}.chkpt'.format(loss=valid_loss))
#                     print('    - [Info] The checkpoint file has been updated.')
#
#
# class TransformerClsModel(TransformerModel):
#     def __init__(self, save_path, log_path, embedding, len_max_seq, output_dim,
#                  d_word_vec=512, d_model=512, d_inner=2048,
#                  n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
#                  warmup_step=100, lr=0.01, is_regression=False, threshold=0):
#
#         super().__init__(save_path, log_path)
#         self.model = TransformerCls(embedding=embedding, len_max_seq=len_max_seq,
#                                     d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
#                                     n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
#                                     output_num=output_dim, dropout=dropout)
#
#         self.optimizer = AdamOptimizer(self.model.parameters(), warmup_step, lr_max=lr)
#         self.is_regression = is_regression
#         self.threshold = threshold
#
#     def train_epoch(self, training_data, device, smoothing):
#         ''' Epoch operation in training phase'''
#
#         self.model.train()
#
#         total_loss = 0
#         n_sample_total = 0
#         n_sample_correct = 0
#
#         for batch in tqdm(
#                 training_data, mininterval=1,
#                 desc='  - (Training)   ', leave=False):  # training_data should be a iterable
#
#             # TODO: batch iterable
#             # prepare data
#             index, position, target= map(lambda x: x.to(device), batch)
#
#             # forward
#             self.optimizer.zero_grad()
#             self.logits = self.model(index, position)
#
#
#
#             # backward
#             if self.is_regression:
#                 # activation
#                 self.pred = nn.Sigmoid(self.logits)
#
#                 loss = self.loss_regression(self.logits, target)
#                 n_correct = self.performance_regression(self.logits, target, threshold=self.threshold)
#
#             else:
#                 loss = self.loss_cross_entropy(self.logits, target, smoothing)
#                 n_correct = self.performance_multi(self.logits, target)
#
#             loss.backward()
#
#             # update parameters
#             self.optimizer.step()
#
#             # note keeping
#             total_loss += loss.item()
#
#             non_pad_mask = target.ne(0)
#             n_sample = non_pad_mask.sum().item()
#             n_sample_total += n_sample
#             n_sample_correct += n_correct
#
#         loss_per_word = total_loss / n_sample_total
#         accuracy = n_sample_correct / n_sample_total
#         return loss_per_word, accuracy
#
#     def eval_epoch(self, validation_data, device):
#         ''' Epoch operation in evaluation phase '''
#
#         self.model.eval()
#
#         total_loss = 0
#         n_sample_total = 0
#         n_sample_correct = 0
#
#         with torch.no_grad():
#
#             # prepare data
#             index, position, target = map(lambda x: x.to(device), validation_data)
#
#             # forward
#             logits = self.model(index, position)
#
#             if self.is_regression:
#                 loss = self.loss_regression(logits, target)
#                 n_correct = self.performance_regression(logits, target, threshold=self.threshold)
#
#             else:
#                 loss = self.loss_cross_entropy(logits, target)
#                 n_correct = self.performance_multi(logits, target)
#
#             # note keeping
#             total_loss += loss.item()
#
#             # note keeping
#             total_loss += loss.item()
#
#             non_pad_mask = target.ne(0)
#             n_sample = non_pad_mask.sum().item()
#             n_sample_total += n_sample
#             n_sample_correct += n_correct
#
#         loss_per_word = total_loss / n_sample_total
#         accuracy = n_sample_correct / n_sample_total
#         return loss_per_word, accuracy
