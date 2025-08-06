# Explore how the dcurves package computes its metrics 
import dcurves as dc
import numpy as np
import pandas as pd


y_true = np.array([0, 0, 0, 1, 1])
y_prob = np.array([0.1, 0.2, 0.2, 0.1, 0.5])
thr = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.9])
d = pd.DataFrame({'y_true': y_true, 'model': y_prob})
r = dc.dca(data=d, outcome='y_true', modelnames=['model'], thresholds=thr)

# Check how prevalence, test_pos_rate and tp_rate and fp_rate are calculated
r.loc[r.model == 'model', 'tp'] = r.loc[r.model == 'model'].apply(lambda x: y_true[y_prob >= x.threshold].sum(), axis=1)
r.loc[r.model == 'all', 'tp'] = y_true.sum()
r.loc[r.model == 'none', 'tp'] = 0
r.tp = r.tp.astype(int)

r.loc[r.model == 'model', 'pp'] = r.loc[r.model == 'model'].apply(lambda x: (y_prob >= x.threshold).sum(), axis=1)
r.loc[r.model == 'all', 'pp'] = len(y_true)
r.loc[r.model == 'none', 'pp'] = 0
r.pp = r.pp.astype(int)

r.loc[r.model == 'model', 'fp'] = r.loc[r.model == 'model'].apply(lambda x: (~y_true.astype(bool))[y_prob >= x.threshold].sum(), axis=1)
r.loc[r.model == 'all', 'fp'] = r.loc[r.model == 'all', 'pp'] - r.loc[r.model == 'all', 'tp']
r.loc[r.model == 'none', 'fp'] = 0
r.fp = r.fp.astype(int)

all(r.prevalence == y_true.sum() / r.n)
all(r.test_pos_rate == r.pp / r.n)
all(r.tp_rate == r.tp / r.n)
all(r.fp_rate == r.fp / r.n)

r['fpr2'] = r.fp / r.n
test = r.fpr2 - r.fp_rate
all(test.abs() < 1e-10)


# Check how net benefit and net intervention avoided are calculated
r['w'] = r.threshold / (1 - r.threshold)
r['nb'] = r.tp_rate - r.fp_rate * r.w
all(r.net_benefit == r.nb)

r['nb_all'] = r.loc[r.model=='all', 'net_benefit'].iloc[0]

r['ni'] = (r.nb - r.nb_all) / r.w
all(r.net_intervention_avoided == r.ni)


# Note that 
# fp_rate = fp / n
# tp_rate = tp / n
# test_pos_rate = pp / n