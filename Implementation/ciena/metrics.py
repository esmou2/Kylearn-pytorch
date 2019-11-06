import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as pyplot


def results(y_real, y_pred, threshold):
    y_real = np.array(y_real).flatten()
    y_pred = np.array(y_pred)
    y_pred = y_pred >= threshold
    y_pred = y_pred.astype(int)
    prec = metrics.precision_score(y_real, y_pred)
    recall = metrics.recall_score(y_real, y_pred)
    num_alarms = len(np.where(y_real == 1)[0])
    num_alarms_correct = len(np.where((y_real == 1) & (y_pred == 1))[0])
    num_pos_preds = len(np.where(y_pred == 1)[0])

    mylist = [y_real.shape[0], num_alarms, num_alarms_correct, num_pos_preds, prec, recall]
    df = pd.DataFrame([mylist], columns = ['num examples','num alarms', 'num_alarms_correct', 'num pos predictions', 'precision', 'recall'])
    return df
