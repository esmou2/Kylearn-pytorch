from Dataloader.reactionattention import ReactionDataloader
from Models.shuffleselfattention import ShuffleSelfAttentionModel
import torch.nn as nn

dataloader = ReactionDataloader('data/', batch_size=256, val_size=0.1)
model = ShuffleSelfAttentionModel('models/ckpt-1d-l2', 'logs/log1d-l2', 16, 30, 256, 2, mode='2d',
                                  n_layers=6, n_head=8, n_channel=8, n_vchannel=8, dropout=0.1, use_bottleneck=True, d_bottleneck=128)

model.load_model(
'/home/oem/Projects/Kylearn-pytorch/Implementation/creditcard_fraud_detection/models/ckpt-2d-l2-step-2400_loss-0.00159'
)

# model.train(200, dataloader.train_dataloader(), dataloader.val_dataloader(), device='cuda', save_mode='best', smoothing=False)


pred, real = model.get_predictions(dataloader.test_dataloader(), 'cuda')

import numpy as np
pred_ = np.array(pred)[:, 1]
real = np.array(real).astype(int)
from utils.plot_curves import precision_recall, plot_pr_curve

area, precisions, recalls, thresholds = precision_recall(pred_, real)
plot_pr_curve(recalls, precisions, auc=area)

from utils.plot_curves import auc_roc, plot_roc_curve
auc, fprs, tprs, thresholds = auc_roc(pred_, real)
plot_roc_curve(fprs, tprs, auc)

from Implementation.ciena.metrics import results

df = results(real, np.array(pred).argmax(axis=-1), 0.5)