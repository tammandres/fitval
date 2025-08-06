"""Explore how sklearn computes the precision-recall curve
Code is based on 
https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b7e21201cfffb118934999025fd50cca/sklearn/metrics/_ranking.py#L712
"""
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from fitval.dummydata import linear_data
from sklearn.linear_model import LogisticRegression


x, y_true = linear_data(n=100)
clf = LogisticRegression()
clf.fit(x, y_true)
y_prob = clf.predict_proba(x)[:, 1]

probas_pred = y_prob
pos_label = None
sample_weight = None


# Get false pos and true pos counts for each observed threshold
fps, tps, thresholds = sm._ranking._binary_clf_curve(
    y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight
)

# Get total count of positives
#  For exploration, add 0 at the start
ps = tps + fps
print(ps, len(ps))

ps = np.append([0], ps)
tps = np.append([0], tps)

# Compute precision
#  Initialize the result array with zeros
#  Points where total number of positives is 0, remain 0
precision = np.zeros_like(tps)
print(precision)
print(ps)

np.divide(tps, ps, out=precision, where=(ps != 0))
print(precision)

# Compute recall: number of positives / total number of positives
# Total number of positives is the last element in the tps vector
recall = tps / tps[-1]
print(recall)

# reverse the outputs so recall is decreasing
sl = slice(None, None, -1)

r0 = np.hstack((precision[sl], 1))
r1 = np.hstack((recall[sl], 0))
r2 = thresholds[sl]
