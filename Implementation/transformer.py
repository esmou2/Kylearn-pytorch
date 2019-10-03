from Models.transformer import TransformerClsModel
import torch
import gensim
import pandas as pd
import numpy as np
from utils.padding import padding_data

# Load word2vec model
word2vec = gensim.models.Word2Vec.load('models/word2vec.model')
vector_length = word2vec.vector_size
embedding = torch.FloatTensor(word2vec.wv.vectors)
# Leave the first embedding to 0 for padding
empty_embedding = torch.zeros_like(embedding[0]).unsqueeze(0)
embedding = torch.cat([empty_embedding, embedding])

# Load data
testdata = pd.read_csv('data/sample_index.csv')
testdata['indexes'] = testdata.indexes.map(lambda x: eval(x))
testdata['length'] = testdata.indexes.map(lambda x: len(x))
testdata['position'] = testdata.length.map(lambda x: list(range(0, x)))
# Leave one for padding
max_length = testdata['length'].max()

# padding data and convert to torch Tensor
# Since the word index and the position index start from 0, we pad them with -1 to distinguish padding
# Then add 1
input = torch.from_numpy(padding_data(testdata['indexes'], max_length, padding_value=-1)+1)
position = torch.from_numpy(padding_data(testdata['position'], max_length, padding_value=-1)+1)
label = torch.from_numpy(testdata['is_vulnerable'].values.reshape(-1, 1))


# Implement model
model = TransformerClsModel('models/', 'logs/', embedding, len_max_seq=max_length, n_tgt_vocab=1,
                            d_word_vec=vector_length, d_model=vector_length, d_inner=512)


device = torch.device('cpu')  # for GPU 'cuda'
model.train(training_data=(input, position, label),
            validation_data=(input, position, label),
            epoch=10, device=device, smoothing=True, save_mode='all')
