from Dataloader.reactionattention import ReactionDataloader
from Models.reactionattention import ReactionModel_
from pytorch_lightning import Trainer
import torch.nn as nn
from torch.optim import Adam

dataloader = ReactionDataloader('/home/oem/Projects/ciena_hackathon/data/', batch_size=500, val_size=0.01)
# dataloader = ReactionDataloader('/Users/sunjincheng/Desktop/Hackathon/data/allpm_anomaly/', batch_size=1000, val_size=0.01)


model = ReactionModel_('models/ckpt', 'logs', dataloader, n_depth=64, d_bottleneck=128, d_classifier=512, n_layers=3,
                       n_head=8, dropout=0.1, d_output=2, stack='ShuffleSelfAttention',
                       expansion_layer='ChannelWiseConvExpansion'
                       )

model.train(epoch=20, device='cuda', smoothing=False, save_mode='best')

# model.load_model('models/ckpt-loss-0.0860723660073497')
# pred, real = model.predict(dataloader.test_dataloader(), 'cuda')

# import numpy as np
# p = np.argmax(pred, axis=0)

# from utils.plot_curves import precision_recall, plot_pr_curve
#
# area, precisions, recalls, thresholds = precision_recall(pred, real)
# plot_pr_curve(recalls, precisions, auc=area)
