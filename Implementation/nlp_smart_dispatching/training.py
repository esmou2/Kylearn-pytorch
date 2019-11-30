from utils.embeddings import get_embeddings
from Implementation.nlp_smart_dispatching.dataloader import TextualDataloader
from Implementation.nlp_smart_dispatching.metrics import metrics_multi, draw_confusion_matrix
from Models.transformer import TransormerClassifierModel
import numpy as np

# Load data loader
dataloader = TextualDataloader('data/train_set.csv','data/test_set.csv',
                                     batch_size=100, eval_portion=0.2, cut_length=100)

# get word2vec embedding list
embedding, vector_length = get_embeddings('data/CBOW.model', padding=True)

save_name = 'nlp-encoder-100-'
# Implement model
model = TransormerClassifierModel('models/'+save_name, 'logs/'+save_name, embedding=embedding,
                                  max_length=dataloader.max_length,  n_classes=dataloader.n_targets,
                                  d_features=100, d_k=100, d_v=100, d_meta=None, n_layers=6, n_head=8, dropout=0.1,
                                  d_classifier=256, use_bottleneck=True, d_bottleneck=128)

# Training
model.train(200, 0.005, dataloader.train_dataloader(), dataloader.val_dataloader(),
            device='cuda', save_mode='best', smoothing=False, earlystop=False)

# # Load model
# model.load_model('/home/oem/Projects/Kylearn-pytorch/Implementation/nlp_smart_dispatching/models/nlp-encoder-balanced--step-133200_loss-2.67607')
#
# # Evaluation
#
# pred, real = model.get_predictions(dataloader.test_dataloader(), 'cuda')
# pred = np.array(pred).argmax(axis = 1)
# real = np.array(real)
#
# from sklearn.externals import joblib
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.rcParams['figure.figsize'] = [10, 5]
# plt.rcParams['figure.dpi'] = 500
#
# encoder = joblib.load('data/label_encoder.pickle')
# cm, acc = metrics_multi(pred, real)
#
# draw_confusion_matrix(cm, encoder.classes_.tolist(), plt, font_size=0.33)
