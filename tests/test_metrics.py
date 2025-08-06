import pandas as pd
import numpy as np
from fitval.dummydata import linear_data
from sklearn.linear_model import LogisticRegression
from fitval.metrics import all_metrics, global_calibration_metrics, global_discrimination_metrics
from fitval.metrics import roc_data, pr_data, metric_at_risk, metric_at_single_sens, dca_table
import matplotlib.pyplot as plt

x, y_true = linear_data(n=500)
x = np.abs(x)
clf = LogisticRegression(penalty=None)
clf.fit(x, y_true)
y_prob = clf.predict_proba(x)[:, 1]


def test_all_metrics():
    p = all_metrics(y_true, y_prob, x[:, 0])
    fields = [f for f, __ in p.__dataclass_fields__.items()]
    for f in fields:
        print('\n\n{}'.format(f))
        d = getattr(p, f)
        print(d.iloc[0:3])


def test_all_metrics_long():
    p = all_metrics(y_true, y_prob, x[:, 0], format_long=True)
    fields = [f for f, __ in p.__dataclass_fields__.items()]
    for f in fields:
        print('\n\n{}'.format(f))
        d = getattr(p, f)
        print(d)


def test_global_calibration_metrics():
    m = global_calibration_metrics(y_true, y_prob)
    print(m)


def test_global_discrimination_metrics():
    m = global_discrimination_metrics(y_true, y_prob)
    print(m)


def test_roc_data():
    u, v, w = roc_data(y_true, y_prob)
    print(u)
    print(v)
    print(w)

    __, __, w = roc_data(y_true, y_prob, interp_spec=True)
    print(w)


def test_pr_data():
    u, v = pr_data(y_true, y_prob)
    print(u)
    print(v)


def test_metrics_at_risk():
    m = metric_at_risk(y_true, y_prob)
    print(m)

    m = metric_at_risk(y_true, y_prob, [0.1, 0.2])
    print(m)


def test_metric_at_single_sens():
    m = metric_at_single_sens(y_true, y_prob, target_sens=0.9)
    print(m)

    m = metric_at_single_sens(y_true, y_prob, target_sens=0.71)
    print(m)


def test_dca_table():
    r = dca_table(y_true, y_prob, thr=[0.05, 0.1, 0.2])
    print(r[['model', 'threshold', 'tp_rate', 'fp_rate', 'net_benefit', 'net_intervention_avoided']])


explore_dc = False
if explore_dc:
        
    # Explore dc
    #  See net benefit formula at https://github.com/ddsjoberg/dcurves/blob/main/R/test_consequences.R
    #  NB - tp_rate is NOT sensitivity, but TP/n
    #   df_results$net_benefit = 
    #   df_results$tp_rate - df_results$threshold / (1 - df_results$threshold) * df_results$fp_rate - df_results$harm
    #
    #  See net in avoided at https://github.com/ddsjoberg/dcurves/blob/main/R/net_intervention_avoided.R
    #   net_intervention_avoided =
    #   (.data$net_benefit - .data$net_benefit_all) / (.data$threshold / (1 - .data$threshold)) * .env$nper
    #   nper: number to report net interventions for; default is 1
    r = dca_table(y_true, y_prob)
    print(r)
    print(r.columns)
    print(r[['model', 'threshold', 'tp_rate', 'fp_rate']])

    # Note how net benefit is computed
    r['w'] = r.threshold / (1 - r.threshold)
    r['nb'] = r.tp_rate - r.fp_rate * r.w - r.harm

    test = np.abs(r.net_benefit - r.nb) == 0
    print(test.all())

    mask = y_prob >= 0.02
    tp_rate = y_true[mask].sum() / y_true.shape[0]

    print(r[['model', 'threshold', 'tp_rate', 'fp_rate', 'net_benefit', 'nb']])

    # Note how net intervention avoided is computed
    a = r.loc[r.model == 'all'][['threshold', 'net_benefit']].rename(columns={'net_benefit': 'nb_all'})
    r = r.merge(a, how='left')
    r['ni'] = (r.nb - r.nb_all) / r.w

    test = np.abs(r.net_intervention_avoided - r.ni) == 0
    test = test[r.threshold != 0]
    print(test.all())

    cols = ['test_pos_rate', 'tp_rate', 'fp_rate', 'net_benefit', 'nb_all', 'net_intervention_avoided']
    r[cols] *= 100
    cols = ['model', 'threshold'] + cols
    print(r[cols])

    pd.set_option('display.min_rows', 200, 'display.max_rows', 200, 'display.max_columns', 10)
    print(r[['model', 'threshold', 'test_pos_rate', 'tp_rate', 'fp_rate', 'fn_rate']])

    # Thresholded ...
    # The kink is due to the fact that at point 0 nb is set to a different value 
    r = dca_table(y_true, y_prob >= 0.5)
    print(r[['model', 'threshold', 'tp_rate', 'fp_rate']])
    print(r)
    print(r.columns)
    r2 = r.loc[r.model == 'model']
    plt.plot(r2.threshold, r2.net_benefit)
    plt.show()