import pandas as pd

pd.options.display.max_rows = 999

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['font.size'] = 15

import numpy as np


class LabelEncoder:
    def __init__(self):
        self.dict = {}
        self.inverse_dict = {}

    def fit(self, indexes):
        self.dict = {v: i for i, v in enumerate(indexes)}
        self.inverse_dict = {i: v for i, v in enumerate(indexes)}
        print('Encode %d classes' % len(self.dict))

    def transform(self, x):
        return [self.dict[v] for v in x]

    def inverse_transform(self, x):
        return [self.inverse_dict[v] for v in x]


data = pd.read_csv('../data/8910_split_loc_dpt.csv', encoding='gb18030')

data.fillna('nan', inplace=True)

location_ltp = data['ltp_loc']
location_jieba = data['jieba_loc']
department_ltp = data['ltp_dpt']
department_jieba = data['jieba_dpt']


def plot_barchart(series):
    value_count = series.value_counts()
    value_count.plot(kind='bar')
    plt.show()


def percentage(series):
    percent = series.value_counts() / series.value_counts().sum()
    return percent


def percentile(series, nth, return_rest=True):
    '''
        Arguments:
            series: {Pandas.Series} -- input array
            nth: {float} -- nth percentile
            return_rest {bool} -- whether to return the classes before and after nth
        Returns:
            decoded {list} -- Element of the nth percentile
            index {int} -- index of the nth percentile element
            after_nth {list} -- a list of elements after nth percentile
    '''
    if nth >= 1:
        nth = float(nth) / 100.

    if series.hasnans:
        series.fillna('nan')
    value_count = series.value_counts()
    encoder = LabelEncoder()
    encoder.fit(value_count.index)
    series_encoded = pd.Series(encoder.transform(series.values))
    index = series_encoded.quantile(nth).astype(int)
    decoded = encoder.inverse_transform([index])

    if return_rest == True:
        indexes_before = np.arange(index)
        before_nth = encoder.inverse_transform(indexes_before)
        indexes_after = np.arange(index + 1, series_encoded.quantile(1).astype(int) + 1)
        after_nth = encoder.inverse_transform(indexes_after)
        return decoded, index, before_nth, after_nth
    return decoded, index,


# print(percentile(department_ltp, 90))
