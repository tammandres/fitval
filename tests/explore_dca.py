# Explore how the dcurves package computes its metrics 
# NB - with different combination of packages dcurves may compute wrong numbers for fp_rate, as in old requirements.txt
# but the current requirements.txt is OK.
import dcurves as dc
import numpy as np
import pandas as pd

y_true = np.array([0, 0, 0, 1, 1])
y_prob = np.array([0.1, 0.2, 0.2, 0.1, 0.5])
thr = np.array([0., 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.9])
d = pd.DataFrame({'y_true': y_true, 'model': y_prob})
r = dc.dca(data=d, outcome='y_true', modelnames=['model'], thresholds=thr)
r

# Check how prevalence, test_pos_rate and tp_rate are calculated
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

test = (r.fp_rate - r.fp / r.n).abs() < 1e-10
all(test)

# Check how net benefit and net intervention avoided are calculated
r['w'] = r.threshold / (1 - r.threshold)
r['nb'] = r.tp_rate - r.fp_rate * r.w
all(r.net_benefit == r.nb)

nb_all = r.loc[r.model == 'all', 'net_benefit'].to_numpy()
r['ni'] = 0.
r.loc[r.model == 'model', 'ni'] = (r.loc[r.model == 'model', 'nb'] - nb_all) / r.w
r.loc[r.model == 'none', 'ni'] = (r.loc[r.model == 'none', 'nb'] - nb_all) / r.w
r.loc[r.model == 'all', 'ni'] = (r.loc[r.model == 'all', 'nb'] - nb_all) / r.w
all(r.net_intervention_avoided == r.ni)
test = (r.net_intervention_avoided - r.ni).abs().dropna() < 1e-10
all(test)


"""
Notes

See net benefit formula at https://github.com/ddsjoberg/dcurves/blob/main/R/test_consequences.R
NB - tp_rate is NOT sensitivity, but TP/n
df_results$net_benefit = 
    df_results$tp_rate - df_results$threshold / (1 - df_results$threshold) * df_results$fp_rate - df_results$harm

See net in avoided at https://github.com/ddsjoberg/dcurves/blob/main/R/net_intervention_avoided.R
net_intervention_avoided =
    (.data$net_benefit - .data$net_benefit_all) / (.data$threshold / (1 - .data$threshold)) * .env$nper
    nper: number to report net interventions for; default is 1


In dcurves R package, the fp_rate is calculated as ...
https://github.com/ddsjoberg/dcurves/blob/main/R/test_consequences.R
fp_rate =
    mean(risk[outcome == "FALSE"] >= .data$threshold) * (1 - .data$pos_rate) %>%
    unname(),
)
So first, it calculates the proportion of negative observations that are predicted as positive:
fp/(fp + tn)
Then it multiplies this by 1 - prevalence.
(1 - prev) = (1 - (tp + fn)/n) = (tp + fp + tn + fn - tp - fn)/n = (fp + tn)/n
fp/(fp + tn) * (fp + tn)/n = fp/n

The code in python package may cause issues in some pandas versions maybe, depending on whether
output of value counts is sorted.

# Also, the kink in curves is due to the fact that at point 0 nb is set to a different value?

"""


## Compare to dcurves tutorial
import statsmodels.api as sm

df_cancer_dx = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')

mod1 = sm.GLM.from_formula('cancer ~ famhistory', data=df_cancer_dx, family=sm.families.Binomial())
mod1_results = mod1.fit()
mod1_results.summary()

t = dc.dca(data=df_cancer_dx, outcome='cancer', modelnames=['famhistory'])
delta = t.test_pos_rate - (t.tp_rate + t.fp_rate)
test = (delta.abs() < 1e-10)
test.all()

t['tp_plus_fp'] = t.tp_rate + t.fp_rate

t.loc[t.model == 'none', 'tp_plus_fp'].unique()  # 0 as expected: test none, no positives
t.loc[t.model == 'all', 'tp_plus_fp'].unique()  # prevalence, not expected, should be 1 as all test positive
t.loc[t.model == 'famhistory', 'tp_plus_fp'].unique()  # 0.1533, expected, proportion of patients w family history
t.prevalence.unique()


df_cancer_dx['fam'] = df_cancer_dx.famhistory.astype(int)
df_cancer_dx['fam'].mean()

delta[~test]
t.iloc[100]


mod2 = sm.GLM.from_formula('cancer ~ marker + age + famhistory', data=df_cancer_dx, family=sm.families.Binomial())
mod2_results = mod2.fit()

# Run dca on multivariable model
t2 = dc.dca(
        data=df_cancer_dx,
        outcome='cancer',
        modelnames=['famhistory', 'cancerpredmarker'],
        thresholds=np.arange(0,0.36,0.01)
    )
t2['tp_plus_fp'] = t2.tp_rate + t2.fp_rate
t2 = t2.loc[t2.model=='cancerpredmarker']

## Compare to 