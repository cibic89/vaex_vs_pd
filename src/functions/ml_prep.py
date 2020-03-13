import numpy as np
from sklearn.utils import class_weight
import xgboost as xgb
import os


def define_weights(y_lbls):
    class_weights = np.round(class_weight.compute_class_weight('balanced', np.unique(y_lbls), y_lbls), 3)
    class_weights = dict(zip([0, 1], class_weights))
    cls_wghts = y_lbls.copy()
    for i in cls_wghts.unique():
        cls_wghts.loc[cls_wghts == i] = class_weights[i]
    return cls_wghts.sort_index()


def dmatricise(data, lbl, weights=True, n_threads=os.cpu_count()):
    if weights:
        return xgb.DMatrix(data=data, label=lbl, weight=define_weights(lbl), nthread=n_threads)
    else:
        return xgb.DMatrix(data=data, label=lbl, nthread=n_threads)
