import pandas as pd
import numpy as np
data = pd.read_csv('data/creditcard.csv')

data['time_diff'] = data.Time - data.Time.shift(1, fill_value=0)
data['log_amount'] = np.log10(data.Amount)
data.loc[data['log_amount'] <= 0, 'log_amount'] = 0
data['log_amount'] = np.ceil(data['log_amount'])
log_amount = data['log_amount'].astype(int)
meta = np.eye(log_amount.max()+1)[log_amount]

data['Amount'] = data['Amount'] / data['Amount'].max()
from sklearn.model_selection import train_test_split

X = data[['time_diff', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                                        'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                                        'V13', 'V14', 'V15', 'V16', 'V17',
                                        'V18', 'V19', 'V20', 'V21', 'V22',
                                        'V23', 'V24', 'V25', 'V26', 'V27',
                                        'V28', 'Amount']]
X_train, X_test = train_test_split(X,
                                  test_size = 0.2, random_state = 42)

meta_train, meta_test = train_test_split(meta, test_size = 0.2, random_state = 42)
y_train, y_test = train_test_split(data['Class'], test_size = 0.2, random_state = 42)

np.save('data/X_train.npy', X_train.values)
np.save( 'data/X_test.npy',X_test.values,)
np.save('data/dev_train.npy',meta_train)
np.save('data/dev_test.npy',meta_test)
np.save('data/y_train.npy',y_train.values.reshape([-1,1]) )
np.save('data/y_test.npy',y_test.values.reshape([-1,1]))