import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.padding import padding_data
from sklearn.model_selection import train_test_split


def process_indexes(index_series, max_length=None):
    """
        Argument:
            index_series {Pandas Series} -- A Series of indexes, each element is a string contains the indexes
        Returns:
            idx_seq {Numpy ndarray, shape [n_rows_of_data, max_length]} -- a 2-D array of word indexes, padding with 0
            pos_seq {Numpy ndarray, shape [n_rows_of_data, max_length]} -- a 2-D array of position indexes, padding with 0
            max_length {Int} -- the max length of all lists in the Series
    """
    # string to list
    # index_series = index_series.reset_index()
    # indexes = index_series.map(lambda x: eval(x))
    indexes = index_series
    # get list length
    length = indexes.map(lambda x: len(x))
    # generate position
    position = length.map(lambda x: list(range(0, x)))
    # get the max length
    max_length_ = length.max()
    # indexes {Pandas Series} -- each element is a list of indexes
    # length {Pandas Seires} -- each element is the length of the list of indexes
    # position {Pandas Series} -- each element is a list of positions
    # max_length {Int} -- the max length of all lists in the Series
    # if max_length is None:
    #     max_length = max_length_
    # else:
    #     if max_length < max_length_:
    #         raise Exception(
    #             'Exists index sequence with a length larger than max_length, please check if max_length applies for all of your datasets.')

    # Pad the sequences with -1 to the same length (max_length), for the use of indexing, shift the sequence with 1
    idx_seq = padding_data(indexes, max_length_, padding_value=-1) + 1
    pos_seq = padding_data(position, max_length_, padding_value=-1) + 1

    return idx_seq, pos_seq, max_length_ # max_length


class TransformerClsDataset(Dataset):
    def __init__(self, train_file_path, test_file_path, eval_size=0.1, max_length=None):
        super(TransformerClsDataset, self).__init__()
        # Load training data
        train_set = pd.read_csv(train_file_path)

        # Split eval set

        train_set, eval_set = train_test_split(train_set, test_size=eval_size, random_state=12)

        # generate dataset
        self.idx_seq_train, self.pos_seq_train, max_length = process_indexes(train_set['indexes'], max_length)

        self.target_train = train_set['is_vulnerable'].values.reshape(-1, 1)

        idx_seq_eval, pos_seq_eval, max_length = process_indexes(eval_set['indexes'], max_length)

        self.eval_set = (torch.from_numpy(idx_seq_eval), torch.from_numpy(pos_seq_eval),
                         torch.from_numpy(eval_set['is_vulnerable'].values.reshape(-1, 1)))

        # load test data

        test_set = pd.read_csv(test_file_path)

        # generate dataset
        idx_seq_test, pos_seq_test, _ = process_indexes(test_set['indexes'], max_length)

        self.test_set = (torch.from_numpy(idx_seq_test), torch.from_numpy(pos_seq_test),
                         torch.from_numpy(test_set['is_vulnerable'].values.reshape(-1, 1)))

        self.max_length = max_length

    def __len__(self):
        return len(self.target_train)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (torch.from_numpy(self.idx_seq_train[index, :]),
                  torch.from_numpy(self.pos_seq_train[index, :]),
                  torch.from_numpy(self.target_train[index, :]))

        return sample


