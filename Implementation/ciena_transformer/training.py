from utils.embeddings import get_embeddings
from Implementation.ciena_transformer.dataloader import CienaPortDataloader
from Implementation.ciena_transformer.model import CienaTransformerModel

# Set parameters
# TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/transformer_dataset_99_61/train/'
# TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/transformer_dataset_99_61/test/'
# N_PM = 61
# N_FACILITY = 25
# L_SEQUENCE = 16
# SAVE_NAME = 'ciena-25-devices-'

TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/two_devices_2/train/train_'
TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/two_devices_2/test/test_'
N_PM = 61
N_FACILITY = 2
L_SEQUENCE = 4
SAVE_NAME = 'ciena-OTM2-ETH10G-2-'

# Load data loader
dataloader = CienaPortDataloader(TRAINSET_PATH, TESTSET_PATH, batch_size=100, eval_portion=0.1, max_length=L_SEQUENCE)



# Implement model
model = CienaTransformerModel('models/'+SAVE_NAME, 'logs/'+SAVE_NAME,
                                  d_features=N_PM,d_meta=N_FACILITY, max_length=L_SEQUENCE, d_classifier=256, n_classes=2,
                                  n_layers=6, n_head=8, dropout=0, use_bottleneck=True, d_bottleneck=128)
#
# # Training
model.train(20, dataloader.train_dataloader(), dataloader.val_dataloader(),
            device='cuda', save_mode='best', smoothing=False, earlystop=False)

# Load trained model

# model.load_model('/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/models/ciena-OTM2-ETH10G--step-7500_loss-0.08620')

# Evaluation

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