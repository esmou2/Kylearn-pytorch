from Dataloader.reactionattention import ReactionDataloader
from Models.reactionattention import ReactionModel
import torch.nn as nn

dataloader = ReactionDataloader('data/', batch_size=256, val_size=0.1)
model = ReactionModel('models/ckpt-11d', 'log11d', n_depth=16, d_features=30, d_meta=6, d_classifier=256, d_output=1,
                      threshold=0.5, mode='1d',
                      n_layers=6, n_head=8, dropout=0.1, use_bottleneck=True, d_bottleneck=256)


# model.resume_checkpoint('models/ckpt-loss-0.0008903071458891007')

# model.train(20, dataloader.train_dataloader(), dataloader.val_dataloader(), device='cuda', save_mode='best', smoothing=False)

model.load_model('models/ckpt-11d-loss-0.00106')

pred, real = model.get_predictions(dataloader.test_dataloader(), 'cuda', activation=nn.functional.sigmoid)

from utils.plot_curves import precision_recall, plot_pr_curve

area, precisions, recalls, thresholds = precision_recall(pred, real)
plot_pr_curve(recalls, precisions, auc=area)

from utils.plot_curves import auc_roc, plot_roc_curve
auc, fprs, tprs, thresholds = auc_roc(pred, real)
plot_roc_curve(fprs, tprs, auc)