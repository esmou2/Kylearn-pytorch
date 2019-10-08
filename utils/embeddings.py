import gensim
import torch

def get_embeddings(model_path, padding = False):
    '''
        Arguments:
            model_path {String} -- gensim model file path
            padding {Boolean} -- Whether to expand a 0 vector for padding
        Returns:
            embedding {Tensor, shape [n_vocabulary, embedding length]} -- word embedding vectors
            vector_length {Int} -- embedding length
    '''
    # Load word2vec model
    word2vec = gensim.models.Word2Vec.load(model_path)
    vector_length = word2vec.vector_size
    embedding = torch.FloatTensor(word2vec.wv.vectors)
    # Leave the first embedding to 0 for padding
    if padding:
        empty_embedding = torch.zeros_like(embedding[0]).unsqueeze(0)
        embedding = torch.cat([empty_embedding, embedding])

    return embedding, vector_length

def one_hot_embedding(index, num_classes):
    diag = torch.eye(num_classes).byte()
    index = index.view(-1).tolist()
    return diag[index]