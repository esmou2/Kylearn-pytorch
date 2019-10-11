from Dataloader.reactionattention import ReactionDataloader
from Modules.reactionattention import ReactionAttentionStack, SelfAttentionStack, AlternateStack, ParallelStack
from Models.reactionattention import ReactionModel_
from pytorch_lightning import Trainer
import torch.nn as nn
from torch.optim import Adam
dataloader = ReactionDataloader('/home/oem/Projects/ciena_hackathon/data/', 1000, 0.01)


model = ReactionModel_('models/ckpt', 'logs', dataloader, Adam, ReactionAttentionStack, d_reactant=64, d_bottleneck=128, d_classifier=512,
                               d_output=2, n_layers=6, )

# model.train(epoch=20, device='cuda', smoothing=False, save_mode='best')

# model = ReactionAttentionStack(10, 10, 10, 10, 10, 10)
# model = nn.DataParallel(model)
# para = model.parameters()
# for i in para:
#     print(i.is_cuda)


model.load_model('models/ckpt-loss-0.0011362792271045802')
pred = model.predict(dataloader.test_dataloader(), 'cuda')

# import numpy as np
# p = np.argmax(pred, axis=0)