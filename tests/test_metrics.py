import pandas as pd
import numpy as np
from fitval.dummydata import dummy_fit_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from fitval.metrics import basic_metrics, all_metrics, global_calibration_metrics, global_discrimination_metrics
from fitval.metrics import roc_data, pr_data, metric_at_risk, metric_at_single_sens, dca_table, pr_gain
from fitval.metrics import metric_at_sens, metric_at_fit_and_mod_threshold, metric_at_fit_sens


# Dummy data
df = dummy_fit_data(n=1000)

# Dummy model
rng = np.random.default_rng(seed=42)
X, y_true = df.drop(labels=['y_true'], axis=1), df.y_true
y_true = y_true.to_numpy()
clf = LogisticRegression(penalty=None)
clf.fit(X, y_true)
y_prob = clf.predict_proba(X)[:, 1]
fit_val = X.fit_val.to_numpy()
roc_auc_score(y_true, y_prob)


def test_basic_metrics():
    """Note
    pp = tp + fp  # total number of positive tests ("predicted positive")
    pn = tn + fn  # total number of negative tests ("predicted negative")
    npos = tp + fn  # number of patients with cancer
    nneg = tn + fp  # number of patients without cancer
    """

    # Compute basic diagnostic metrics
    thr = 0.2
    pred = (y_prob >= thr).astype(int)
    m = basic_metrics(y_true, pred)

    # Compute basic diagnostic metrics again from their definition
    n = len(y_true)
    npos = y_true.sum()
    nneg = len(y_true) - npos
    
    tp = pred[y_true == 1].sum()
    fp = pred[y_true == 0].sum()
    tn = nneg - fp
    fn = npos - tp

    pp = tp + fp
    pn = tn + fn

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    sens = tp / npos
    spec = tn / nneg

    pp1000 = pp / n * 1000
    pn1000 = pn / n * 1000
    tp1000 = tp / n * 1000
    fp1000 = fp / n * 1000
    tn1000 = tn / n * 1000
    fn1000 = fn / n * 1000
    
    assert int(tp) == int(m.tp)
    assert int(fp) == int(m.fp)
    assert int(tn) == int(m.tn)
    assert int(fn) == int(m.fn)
    assert abs(sens - m.sens) < 1e-10
    assert abs(spec - m.spec) < 1e-10
    assert abs(ppv - m.ppv) < 1e-10
    assert abs(npv - m.npv) < 1e-10
    assert abs(pp1000 - m.pp1000) < 1e-10
    assert abs(pn1000 - m.pn1000) < 1e-10
    assert abs(tp1000 - m.tp1000) < 1e-10
    assert abs(fp1000 - m.fp1000) < 1e-10
    assert abs(tn1000 - m.tn1000) < 1e-10
    assert abs(fn1000 - m.fn1000) < 1e-10


def test_all_metrics():
    p = all_metrics(y_true, y_prob, fit_val)
    fields = [f for f, __ in p.__dataclass_fields__.items()]
    for f in fields:
        print('\n\n{}'.format(f))
        d = getattr(p, f)
        print(d.iloc[0:3])


def test_all_metrics_thr_mod():
    p = all_metrics(y_true, y_prob, fit_val, thr_fit=[2, 10, 100], thr_mod=[0.1, 0.01, 0.5])
    print(p.thr_fit_mod)


def test_all_metrics_long():
    p = all_metrics(y_true, y_prob, fit_val, format_long=True)
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


def test_pr_gain():
    g = pr_gain(y_true, y_prob, fit_val)
    print(g)


def test_metric_at_risk():
    m = metric_at_risk(y_true, y_prob)
    print(m)

    m = metric_at_risk(y_true, y_prob, [0.1, 0.2])
    print(m)


def test_metric_at_sens():
    m = metric_at_sens(y_true, y_prob)
    print(m)

    m = metric_at_sens(y_true, y_prob, [0.33, 0.95])
    print(m)


def test_metric_at_fit_and_mod_thr():
    m = metric_at_fit_and_mod_threshold(y_true, y_prob, fit_val)
    print(m)

    m = metric_at_fit_and_mod_threshold(y_true, y_prob, fit_val, thr_fit=[100], thr_mod=[0.9])
    print(m)


def test_metric_at_fit_sens():
    m = metric_at_fit_sens(y_true, y_prob, fit_val)
    print(m)

    m = metric_at_fit_sens(y_true, y_prob, fit_val, thr_fit=[1, 7])
    print(m)


def test_metric_at_single_sens():
    m = metric_at_single_sens(y_true, y_prob, target_sens=0.9)
    print(m)

    m = metric_at_single_sens(y_true, y_prob, target_sens=0.71)
    print(m)


def test_dca_table():
    r = dca_table(y_true, y_prob, thr=[0.05, 0.1, 0.2])
    print(r[['model', 'threshold', 'tp_rate', 'fp_rate', 'net_benefit', 'net_intervention_avoided']])

    # Compute key quantities in DCA table from their definition
    r.loc[r.model == 'model', 'tp_comp'] = r.loc[r.model == 'model'].apply(lambda x: y_true[y_prob >= x.threshold].sum(), axis=1)
    r.loc[r.model == 'all', 'tp_comp'] = y_true.sum()
    r.loc[r.model == 'none', 'tp_comp'] = 0

    r.loc[r.model == 'model', 'pp_comp'] = r.loc[r.model == 'model'].apply(lambda x: (y_prob >= x.threshold).sum(), axis=1)
    r.loc[r.model == 'all', 'pp_comp'] = len(y_true)
    r.loc[r.model == 'none', 'pp_comp'] = 0

    r.loc[r.model == 'model', 'fp_comp'] = r.loc[r.model == 'model'].apply(lambda x: (~y_true.astype(bool))[y_prob >= x.threshold].sum(), axis=1)
    r.loc[r.model == 'all', 'fp_comp'] = r.loc[r.model == 'all', 'pp'] - r.loc[r.model == 'all', 'tp']
    r.loc[r.model == 'none', 'fp_comp'] = 0

    r['prevalence_comp'] = y_true.sum() / r.n
    r['test_pos_rate_comp'] = r.pp / r.n
    r['tp_rate_comp'] = r.tp / r.n
    r['fp_rate_comp'] = r.fp / r.n
    r['odds'] = r.threshold / (1 - r.threshold)
    r['net_benefit_comp'] = r.tp_rate - r.fp_rate * r.odds
    r['pn_comp'] = r.n - r.pp
    r['tn_comp'] = r.n * (1 - r.prevalence) - r.fp
    r['fn_comp'] = r.n * r.prevalence - r.tp
    r['net_intervention_avoided_comp'] = r.tn / r.n - r.fn / r.n * (1 / r.odds)

    cols = ['prevalence', 'test_pos_rate', 'tp_rate', 'fp_rate', 'net_benefit', 'net_intervention_avoided',
            'tp', 'fp', 'tn', 'fn', 'pn']
    for c in cols:
        assert all((r[c] - r[c + '_comp']).abs() < 1e-10)
