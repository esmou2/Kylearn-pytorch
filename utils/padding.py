import numpy as np

def padding_data(data, max_length, padding_value):
    '''
        padding a list of lists to max_length, and return a 2D list
        Arguments:
            data {Pandas Series} -- sequence to be padded, each element is a list of word index
            max_length {Int} -- the max length of the sequences
            padding_value {Int} -- the value to fill in
        Returns:
            data {2-D Numpy ndarray} -- a 2-D array with shape [number of sequences, max_length]
    '''

    def padding(example):
        '''padding the lists to the same length'''
        seq_length = len(example)
        return np.pad(example, (0, max_length - seq_length), constant_values=padding_value, mode='constant')
    data = data.map(padding)
    data = np.concatenate(data.values)
    data = data.reshape(-1, max_length) # Squeeze the lists to one list, and reshape

    return data
