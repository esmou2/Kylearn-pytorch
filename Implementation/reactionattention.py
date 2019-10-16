from Dataloader.reactionattention import ReactionDataloader
from Modules.reactionattention import ReactionAttentionStack, SelfAttentionStack, AlternateStack, ParallelStack
from Models.reactionattention import ReactionModel_
from pytorch_lightning import Trainer
import torch.nn as nn
from torch.optim import Adam
dataloader = ReactionDataloader('/home/oem/Projects/ciena_hackathon/data/', 1000, 0.01)


model = ReactionModel_('models/ckpt', 'logs', dataloader, Adam, ReactionAttentionStack, d_reactant=64, d_bottleneck=128, d_classifier=512,
                               d_output=1, n_layers=6, threshold=0.5)

model.train(epoch=20, device='cuda', smoothing=False, save_mode='best')

# model = ReactionAttentionStack(10, 10, 10, 10, 10, 10)
# model = nn.DataParallel(model)
# para = model.parameters()
# for i in para:
#     print(i.is_cuda)

#
# model.load_model('models/ckpt-loss-0.0860723660073497')
pred, real = model.predict(dataloader.test_dataloader(), 'cuda')

# import numpy as np
# p = np.argmax(pred, axis=0)

from utils.plot_curves import precision_recall, plot_pr_curve
area, precisions, recalls, thresholds = precision_recall(pred, real)
plot_pr_curve(recalls,precisions, auc=area)