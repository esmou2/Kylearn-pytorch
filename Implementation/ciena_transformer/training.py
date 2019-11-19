from utils.embeddings import get_embeddings
from Implementation.ciena_transformer.dataloader import CienaPortDataloader
from Implementation.ciena_transformer.model import CienaTransformerModel

param = 1
train_path = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/'
# Load data loader
dataloader = CienaPortDataloader(train_path, train_path, batch_size=100, eval_portion=0.2, max_length=4)



save_name = 'ciena-'
# Implement model
model = CienaTransformerModel('models/'+save_name, 'logs/'+save_name,
                                  d_features=65,d_meta=2, max_length=4, d_classifier=256, n_classes=2,
                                  n_layers=6, n_head=8, dropout=0, use_bottleneck=True, d_bottleneck=128)

# Training
model.train(20, dataloader.train_dataloader(), dataloader.val_dataloader(),
            device='cuda', save_mode='best', smoothing=False, earlystop=False)

# # Evaluation
#
# pred, real = model.get_predictions(dataloader.test_dataloader(), 'cuda')
#
# import numpy as np
# pred_ = np.array(pred)[:, 1]
# real = np.array(real).astype(int)
# from utils.plot_curves import precision_recall, plot_pr_curve
#
# area, precisions, recalls, thresholds = precision_recall(pred_, real)
# plot_pr_curve(recalls, precisions, auc=area)
#
# from utils.plot_curves import auc_roc, plot_roc_curve
# auc, fprs, tprs, thresholds = auc_roc(pred_, real)
# plot_roc_curve(fprs, tprs, auc)
#
# from Implementation.ciena.metrics import results
#
# df = results(real, np.array(pred).argmax(axis=-1), 0.5)