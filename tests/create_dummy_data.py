import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from fitval.dummydata import dummy_fit_data, induce_missing
from constants import PROJECT_ROOT


test_path = PROJECT_ROOT / 'tests' / 'test_data'
test_path.mkdir(exist_ok=True, parents=True)


# Create dummy data
df_test = dummy_fit_data(n=5000, random_state=42)
df_train = dummy_fit_data(n=5000, random_state=256)

# Fit dummy models
x_train, y_train = df_train.drop(labels=['y_true'], axis=1), df_train.y_true
x_test, y_test = df_test.drop(labels=['y_true'], axis=1), df_test.y_true

mask = x_test.columns != 'ind_gender_M'
x_test_tf = x_test.copy()
x_test_tf.loc[:, mask] = np.log(x_test_tf.loc[:, mask] + 1)
x_train_tf = x_train.copy()
x_train_tf.loc[:, mask] = np.log(x_train_tf.loc[:, mask] + 1)

clf = LogisticRegression()
clf.fit(x_train_tf, y_train)
pred_train = clf.predict_proba(x_train_tf)[:, 1]
pred_test = clf.predict_proba(x_test_tf)[:, 1]
roc_auc_score(y_train, pred_train)
roc_auc_score(y_test, pred_test)

clf2 = LogisticRegression()
clf2.fit(x_train_tf[['fit_val', 'age', 'ind_gender_M']], y_train)
pred_train2 = clf2.predict_proba(x_train_tf[['fit_val', 'age', 'ind_gender_M']])[:, 1]
pred_test2 = clf2.predict_proba(x_test_tf[['fit_val', 'age', 'ind_gender_M']])[:, 1]
roc_auc_score(y_train, pred_train2)
roc_auc_score(y_test, pred_test2)

# Save data with model predictions
dfpred = df_test[['y_true', 'fit_val']].copy()
dfpred['y_pred1'] = pred_test
dfpred['y_pred2'] = pred_test2
path_pred = test_path / 'pred.csv'
dfpred.to_csv(path_pred, index=False)

# Save data with predictor variables and no model predictions
path_xy = test_path / 'xy.csv'
df_test.to_csv(path_xy, index=False)

# Save data with predictor variables and no predictions, where some values are missing
df_test_mis = induce_missing(x_test)
df_test_mis['y_true'] = y_test 
path_xy_mis = test_path / 'xy_mis.csv'
df_test_mis.to_csv(path_xy_mis, index=False)