from Dataloader.reactionattention import ReactionDataloader
from Modules.reactionattention import ReactionAttentionStack, SelfAttentionStack, AlternateStack, ParallelStack
from Models.reactionattention import ReactionModel
from pytorch_lightning import Trainer

dataloader = ReactionDataloader('/Users/sunjincheng/Desktop/Hackathon/data/allpm_anomaly/', 1000, 0.01)

model = ReactionModel(dataloader, ReactionAttentionStack, d_reactant=64, d_bottleneck=128, d_classifier=512,
                      d_output=1, threshold=0.8)

trainer = Trainer()

trainer.fit(model)