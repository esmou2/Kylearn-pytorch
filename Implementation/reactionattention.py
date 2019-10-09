from Dataloader.reactionattention import ReactionDataloader
from Modules.reactionattention import ReactionAttentionStack, SelfAttentionStack, AlternateStack, ParallelStack
from Models.reactionattention import ReactionModelLightning
from pytorch_lightning import Trainer

dataloader = ReactionDataloader('/home/oem/Projects/ciena_hackathon/data/', 1000, 0.01)

model = ReactionModelLightning(dataloader, ReactionAttentionStack, d_reactant=64, d_bottleneck=128, d_classifier=512,
                               d_output=1, threshold=0.8)

trainer = Trainer(gpus=[1])

trainer.fit(model)