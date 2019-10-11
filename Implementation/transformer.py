import torch
import pandas as pd
from utils.embeddings import get_embeddings
from Models.transformer import TransformerClsModel
from Dataloader.transformer import TransformerClsDataset, TransformerClsDataLoader

# Load data
dataset = TransformerClsDataset(train_file_path='data/sample_index.csv', test_file_path='data/sample_index.csv',
                                eval_size=0.1, max_length=207)
data_loader = TransformerClsDataLoader(dataset, batch_size=100)
# get word2vec embedding list
embedding, vector_length = get_embeddings('models/word2vec.model', padding=True)

# Implement model
model = TransformerClsModel('models/', 'logs/', embedding, len_max_seq=207, output_dim=2,
                            d_word_vec=vector_length, d_model=vector_length, d_inner=512)

device = torch.device('cpu')  # for GPU 'cuda'
model.train(training_data=data_loader.loader,
            validation_data=dataset.eval_set,
            epoch=10, device=device, smoothing=True, save_mode='all')
