import torch
import torch.nn.functional as F
from utils.embeddings import one_hot_embedding

def cross_entropy_loss(logits, real, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    real = real.contiguous().view(-1).long()

    if smoothing:
        eps = 0.1
        n_class = logits.size(1)
        pred = logits.log_softmax(dim=1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(eps / (n_class - 1))
            true_dist.scatter_(1, real.data.unsqueeze(1), 1 - eps)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))
    else:
        # loss = F.cross_entropy(logits, real, ignore_index=0, reduction='sum')
        loss = F.cross_entropy(logits, real)
        return loss


def mse_loss(pred, real):
    loss = F.mse_loss(pred, real)
    return loss
