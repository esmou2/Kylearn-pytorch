import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sb

def metrics_multi(y_pred, y_test):
    cm = confusion_matrix(np.array(y_test).flatten(), np.array(y_pred).flatten())
    print(classification_report(np.array(y_test).flatten(), np.array(y_pred).flatten()))
    acc = accuracy_score(y_test, y_pred)
    return cm, acc

def draw_confusion_matrix(cm, labels, plt, x_rotation=90, y_rotation=0, font_size=0.33, precision=False):

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif (c == 0):
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)

    cm.index.name = 'True Label'
    cm.columns.name = 'Predict Label'

    sb.set(font_scale=font_size)
    sb.set(font='SimHei')
    sb.heatmap(cm, annot=annot, fmt='', cmap='Blues')
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)

    plt.show()