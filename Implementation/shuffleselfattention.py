from Dataloader.reactionattention import ReactionDataloader
from Models.shuffleselfattention import ShuffleSelfAttentionModel
from pytorch_lightning import Trainer
import torch.nn as nn
from torch.optim import Adam

dataloader = ReactionDataloader('/home/oem/Projects/ciena_hackathon/data/', batch_size=256, val_size=0.01)
# dataloader = ReactionDataloader('/Users/sunjincheng/Desktop/Hackathon/data/allpm_anomaly/', batch_size=1000, val_size=0.01)


model = ShuffleSelfAttentionModel('models/ckpt', 'logs', 16, 211, 256, 1, 0.5, mode='2d',
                                  n_layers=4, n_head=8, n_channel=8, n_vchannel=8, dropout=0.1, use_bottleneck=True, d_bottleneck=128)


model.train(20, dataloader.train_dataloader(), dataloader.val_dataloader(), device='cuda', save_mode='best', smoothing=False)

# model.load_model('models/ckpt-loss-0.0860723660073497')
# pred, real = model.predict(dataloader.test_dataloader(), 'cuda')

# import numpy as np
# p = np.argmax(pred, axis=0)

# from utils.plot_curves import precision_recall, plot_pr_curve
#
# area, precisions, recalls, thresholds = precision_recall(pred, real)
# plot_pr_curve(recalls, precisions, auc=area)
