"""additional check of results"""

import pandas as pd
from pathlib import Path
from fitval.models import get_model
from fitval.metrics import core_metric_at_threshold, metric_at_fit_sens
from constants import PROJECT_ROOT
import numpy as np
import matplotlib.pyplot as plt


# Settings
thr_colofit = 0.0064
fu = 180
B = 0
seed = 42

data_path = Path('C:\\Users\\5lJC\Desktop\\dataprep_fitml_and_fitval\\data-colofit_fu-' + str(fu))

df = pd.read_csv(data_path / 'data_matrix.csv')
fit = pd.read_csv(data_path / 'fit.csv')
fit.fit_date = pd.to_datetime(fit.fit_date)
print(df.shape)
print(df.groupby('crc').size())

(df.fit_val >= 10).mean()

precovid = fit.loc[fit.fit_date < '2020-03-01']
covid = fit.loc[(fit.fit_date >= '2020-03-01') & (fit.fit_date < '2021-05-01')]
post1 = fit.loc[(fit.fit_date >= '2021-05-01') & (fit.fit_date < '2022-01-01')]
post2 = fit.loc[(fit.fit_date >= '2022-01-01') & (fit.fit_date < '2022-07-01')]
post3 = fit.loc[(fit.fit_date >= '2022-07-01') & (fit.fit_date < '2023-01-01')]
post4 = fit.loc[(fit.fit_date >= '2023-01-01') & (fit.fit_date < '2023-07-01')]

date_thr = fit.loc[fit.stool_pot == 1].fit_date.min()
fitsub = fit.loc[(fit.stool_pot == 0) & (fit.fit_date >= date_thr)]

post3buff = fitsub.loc[(fitsub.fit_date >= '2022-07-01') & (fitsub.fit_date < '2023-01-01')]
post4buff = fitsub.loc[(fitsub.fit_date >= '2023-01-01') & (fitsub.fit_date < '2023-07-01')]

fit_data = {'all': fit,
            'precovid': precovid, 
            'covid': covid, 
            'post1': post1,
            'post2': post2, 
            'post3': post3,
            'post4': post4,
            #'post3-buffcomm': post3buff,
            #'post4-buffcomm': post4buff,
            }

if fu == 365:  # too small n
    fit_data.pop('post4')
    fit_data.pop('post4-buffcomm')

matrix = {}
for label, f in fit_data.items():
    dfsub = df.loc[df.patient_id.isin(f.patient_id)]
    matrix[label] = dfsub
    print(label, dfsub.shape[0], dfsub.crc.sum())




model_name = 'nottingham-cox'
model = get_model(model_name)

gain = pd.DataFrame()
gain_boot = pd.DataFrame()
rule_out = pd.DataFrame()

rng = np.random.default_rng(seed=seed)
labels = list(matrix.keys())

i = 2
label = labels[i]
print('\n----', label, matrix[label].shape)


### Check automatic calculation against manual

# Estimate model threshold on previous data subset
if i > 1:
    label_prev = labels[i - 1]
    df_prev = matrix[label_prev]
    y_prev, fit_prev = df_prev.crc.to_numpy(), df_prev.fit_val.to_numpy()
    pred_prev = model(df_prev)
    m, __ = metric_at_fit_sens(y_prev, pred_prev, fit_prev, thr_fit = [10])
    thr_mod = m.loc[m.model == 'model', 'thr'].item()
else:
    thr_mod = None

# Performance on original data
dfsub = matrix[label]
y_true, fit_val = dfsub.crc.to_numpy(), dfsub.fit_val.to_numpy()
y_pred = model(dfsub)


## External threshold
g = core_metric_at_threshold(y_true, y_pred, fit_val, thr_mod = thr_colofit, thr_fit = 10, long_format=False)
g = pd.melt(g, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
g['thr_method'] = 'external'

pos_mod = (y_pred >= thr_colofit).astype(int)
pos_fit = (fit_val >= 10).astype(int)
pp_mod, pp_fit = pos_mod.sum(), pos_fit.sum()
r = (pp_mod - pp_fit) / pp_fit
sens_mod = y_true[pos_mod == 1].sum() / y_true.sum()
sens_fit = y_true[pos_fit == 1].sum() / y_true.sum()
print(pp_mod, pp_fit, r, sens_mod, sens_fit)
g = g.set_index('metric_name')['metric_value']
print(g['pp_mod'], g['pp_fit'], g['proportion_reduction_tests'], g['sens_mod'], g['sens_fit'])


## Local threshold using current data subset
m_curr, __ = metric_at_fit_sens(y_true, y_pred, fit_val, thr_fit = [10])
thr_mod_curr = m_curr.loc[m_curr.model == 'model', 'thr'].item()
g_curr = core_metric_at_threshold(y_true, y_pred, fit_val, thr_mod = thr_mod_curr, thr_fit = 10, long_format=False)
g_curr = pd.melt(g_curr, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
g_curr['thr_method'] = 'local-current'

pos_mod = (y_pred >= thr_mod_curr).astype(int)
pos_fit = (fit_val >= 10).astype(int)
pp_mod, pp_fit = pos_mod.sum(), pos_fit.sum()
r = (pp_mod - pp_fit) / pp_fit
sens_mod = y_true[pos_mod == 1].sum() / y_true.sum()
sens_fit = y_true[pos_fit == 1].sum() / y_true.sum()
print(pp_mod, pp_fit, r, sens_mod, sens_fit)

print(pp_mod, pp_fit, r)
g = g_curr.set_index('metric_name')['metric_value']
print(g['pp_mod'], g['pp_fit'], g['proportion_reduction_tests'], g['sens_mod'], g['sens_fit'])


## Check that the optimal threshold is maximal
print(sens_mod)
pos_mod = (y_pred >= (thr_mod_curr + 0.0001)).astype(int)
pos_fit = (fit_val >= 10).astype(int)
pp_mod, pp_fit = pos_mod.sum(), pos_fit.sum()
r = (pp_mod - pp_fit) / pp_fit
sens_mod = y_true[pos_mod == 1].sum() / y_true.sum()
sens_fit = y_true[pos_fit == 1].sum() / y_true.sum()
print(pp_mod, pp_fit, r, sens_mod, sens_fit)


## Check calibration slope and intercept
from sklearn.linear_model import LogisticRegression
pos_mod = (y_pred >= thr_mod_curr).astype(int)
pos_fit = (fit_val >= 10).astype(int)
pp_mod, pp_fit = pos_mod.sum(), pos_fit.sum()
r = (pp_mod - pp_fit) / pp_fit
sens_mod = y_true[pos_mod == 1].sum() / y_true.sum()
sens_fit = y_true[pos_fit == 1].sum() / y_true.sum()

y_prob = y_pred

event_rate = y_true.mean() * 100
mean_risk = y_prob.mean() * 100
oe_ratio = event_rate / mean_risk
print(event_rate, mean_risk, oe_ratio)

clf = LogisticRegression(penalty=None)
clf.fit(np.log(y_prob / (1 - y_prob)).reshape(-1, 1), y_true)
np.round(clf.coef_, 3)  # 1.05
np.round(clf.intercept_, 3)  # 0.67

logit_prob = np.log((y_prob) / (1 - y_prob))

import statsmodels.api as sm

X = pd.DataFrame({'intercept': 1, 'logit': logit_prob})

logistic_model = sm.GLM(y_true, X.values, family=sm.families.Binomial())
result = logistic_model.fit()
result.summary()

X_int = X[['intercept']]

logistic_model = sm.GLM(y_true, X_int.values, family=sm.families.Binomial(), offset=logit_prob)
result = logistic_model.fit()
result.summary()


out_path = PROJECT_ROOT / 'results' / 'check-cal-in-r'
out_path.mkdir(exist_ok=True, parents=True)

df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'label': label, 'model': model_name})
df.to_csv(out_path / 'predictions.csv')


from fitval.metrics import lowess_calibration_curve
r = lowess_calibration_curve(y_true, y_prob)
plt.plot(r.prob_pred, r.prob_true)
plt.plot([0, 1], [0, 1], color='red')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()




