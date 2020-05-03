from utils.embeddings import get_w2v_embeddings
from Implementation.ciena_transformer.dataloader import CienaPortDataloader
from Implementation.ciena_transformer.model import CienaTransformerModel

# Set parameters
# TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/transformer_datase_99_61_OTM2_ETH10G/train/'
# TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/transformer_datase_99_61_OTM2_ETH10G/test/'
# N_PM = 61
# N_FACILITY = 2
# L_SEQUENCE = 4
# SAVE_NAME = 'ciena-OTM2-ETH10G-new-'

# TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/transformer_datase_99_61_all/train/'
# TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/transformer_datase_99_61_all/test/'
# N_PM = 61
# N_FACILITY = 25
# L_SEQUENCE = 16
# SAVE_NAME = 'ciena-all-1day'

# TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/transformer_dataset_99_61_1past5future_all/train/train_'
# TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/transformer_dataset_99_61_1past5future_all/test/test_'
# N_PM = 61
# N_FACILITY = 25
# L_SEQUENCE = 8
# SAVE_NAME = 'ciena-all-1day'

# TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/p2f5_OTM2-ETH10G/train/train_'
# TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/p2f5_OTM2-ETH10G/test/test_'
# N_PM = 61
# N_FACILITY = 2
# L_SEQUENCE = 4
# SAVE_NAME = 'p2f5_OTM2-ETH10G'
#
# TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/p2f5_OTM2-ETH10G_upsampled/train/train_'
# TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/p2f5_OTM2-ETH10G_upsampled/test/test_'
# N_PM = 61
# N_FACILITY = 2
# L_SEQUENCE = 4
# SAVE_NAME = 'p2f5_OTM2-ETH10G_upsampled2'

# TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/p123f5_OTM2-ETH10G_allpms/train/train_'
# TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/p123f5_OTM2-ETH10G_allpms/test/test_'
# N_PM = 211
# N_FACILITY = 2
# L_SEQUENCE = 6
# SAVE_NAME = 'p123f5_OTM2-ETH10G_allpms'


TRAINSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/p123f5_strats1_uas300_pca99_OTM2-ETH10G_seperated/train/'
TESTSET_PATH = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/data/p123f5_strats1_uas300_pca99_OTM2-ETH10G_seperated/test/'
N_PM = 61
N_FACILITY = 2
L_SEQUENCE = 6
SAVE_NAME = 'p123f5_strats1_uas300_pca99_OTM2-ETH10G_seperated_lr00001'

# Load data loader
dataloader = CienaPortDataloader(TRAINSET_PATH, TESTSET_PATH, batch_size=4096, eval_portion=0.1, max_length=L_SEQUENCE)



# Implement model
model = CienaTransformerModel('models/'+SAVE_NAME, 'logs/'+SAVE_NAME,
                                  d_features=N_PM,d_meta=N_FACILITY, max_length=L_SEQUENCE, d_classifier=256, n_classes=2,
                                  n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, use_bottleneck=True, d_bottleneck=128)

# Training
model.train(1000, dataloader.train_dataloader(), dataloader.val_dataloader(),
            device='cuda', save_mode='best', smoothing=False, earlystop=False)

# # Load trained model
# folder = '/home/oem/Projects/Kylearn-pytorch/Implementation/ciena_transformer/models/'
# model_path = 'p2f5_strats1_uas300_25facilities_pca99-step-200_loss-0.58974'
# model.load_model(folder + model_path)

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

df = results(real, pred_, 0.8)

