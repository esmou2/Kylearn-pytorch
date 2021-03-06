from Implementation.ciena.dataloader import CienaDataloader
from Models.shuffleselfattention import ShuffleSelfAttentionModel
import torch.nn as nn
dataloader = CienaDataloader('data/', 100, val_size=0.1)
model = ShuffleSelfAttentionModel('models/ckpt-1d', 'logs/1d', 16, 102, 256, 2, mode='1d',
                                  n_layers=6, n_head=8, n_channel=8, n_vchannel=8, dropout=0.1, use_bottleneck=True, d_bottleneck=128)


# model.load_model('/home/oem/Projects/Kylearn-pytorch/Implementation/ciena/models/ckpt-2d-step-12700_loss-0.53049')

model.train(400, dataloader.train_dataloader(), dataloader.val_dataloader(),
            device='cuda', save_mode='best', smoothing=False, earlystop=False)


#
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