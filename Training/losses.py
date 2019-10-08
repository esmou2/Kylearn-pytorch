import torch
import torch.nn.functional as F

def cross_entropy_loss(logits, real, smoothing=False):
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


def mse_loss(logits, real):
    z = F.sigmoid(logits)
    loss = F.mse_loss(z, real)

    return loss