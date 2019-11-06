from numpy.lib.stride_tricks import as_strided as strided
import torch.as_strided

def get_sliding_windows(dataFrame, windowSize, return_flatten=False):
    '''
        Arguments:
            dataFrame {Pandas DataFrame} -- dataframe to be strided
            windowSize {Int} -- sliding window size
            return_flatten {bool} -- whether to flatten rows

        Returns:
            output {Numpy ndarray} -- strided data
    '''
    stride_row, stride_col = dataFrame.values.strides
    rows, columns = dataFrame.shape
    output = strided(dataFrame, shape=(rows - windowSize + 1, windowSize, columns),
                     strides=(stride_row, stride_row, stride_col))
    if return_flatten == 1:
        return output.reshape(dataFrame.shape[0] - windowSize + 1, -1)
    else:
        return output