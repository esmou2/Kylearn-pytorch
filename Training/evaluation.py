from utils.embeddings import one_hot_embedding
import torch


def performance_multi(logits, real):
    logits = logits.max(1)[1]
    real = real.contiguous().view(-1)
    non_pad_mask = real.ne(0)
    n_correct = logits.eq(real)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return n_correct


def performance_regression(logits, real, threshold):
    ''' Apply label smoothing if needed '''

    logits = logits.ge(threshold)
    real = real.contiguous().view(-1)
    non_pad_mask = real.ne(0)
    n_correct = logits.eq(real)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return n_correct


def accuracy(pred, real, threshold=None):
    if threshold==None:
        n = real.shape[0]
        pred = pred.argmax(dim=-1).view(n, -1)
        real = real.view(n, -1)
        acc = pred.byte() == real.byte()
        acc = (acc).float().mean()
        return acc
    else:
        acc = (pred > threshold).byte() == real.byte()
        acc = (acc).float().mean()
        return acc



def confusion_matrix(pred, real, threshold=None):
    '''
        For binary or multi class classification
        If pred is probabilities, convert to binary form
    '''
    dim = real.shape[-1]
    if dim == 1:
        if threshold != None:
            pred = (pred > threshold).byte()
            pred = one_hot_embedding(pred, 2)
            real = one_hot_embedding(real, 2)
        else:
            pred = one_hot_embedding(pred, real.max(-1))
            real = one_hot_embedding(real, real.max(-1))

    cm = torch.bmm(real.t(), pred)
    return cm


def precision_recall(pred, real, d_output, threshold=None, average='macro', eps = 1e-6):
    n = real.shape[0]
    dim = real.shape[-1]
    if dim == 1:
        if threshold != None:
            pred = (pred > threshold).byte()
            pred = one_hot_embedding(pred, 2)
            real = one_hot_embedding(real, 2)
        else:
            pred = pred.argmax(dim=-1).view(n, -1)
            real = real.view(n, -1)
            pred = one_hot_embedding(torch.argmax(pred, -1).view(n, -1), d_output)

            real = one_hot_embedding(torch.argmax(real, -1).view(n, -1), d_output)


    tp = (pred & real)
    tp_count = tp.sum(0).float()
    fp_count = (pred - tp).sum(0).float() + eps
    fn_count = (real - tp).sum(0).float() + eps
    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)

    if threshold != None:
        return precision, recall, precision, recall

    else:
        if average == 'macro':
            precision_avg = precision.mean()
            recall_avg = precision.mean()

        else:
            precision_avg = tp.sum(0).sum() / n
            recall_avg = precision_avg

    return precision, recall, precision_avg, recall_avg

class Evaluator():
    def __init__(self):
        self.total_loss = 0
        self.total_acc = 0
        self.total_pre = 0
        self.total_rec = 0
        self.batch_counter = 0


    def __call__(self, loss, acc, precision, recall):
        self.total_loss += loss
        self.total_acc += acc
        self.total_pre += precision
        self.total_rec += recall
        self.batch_counter += 1

    def avg_results(self):
        loss_avg = self.total_loss / self.batch_counter
        acc_avg = self.total_acc / self.batch_counter
        pre_avg = self.total_pre / self.batch_counter
        rec_avg = self.total_rec / self.batch_counter
        return loss_avg, acc_avg, pre_avg, rec_avg
