"""Plot FIT timetrends
The code is slow with Wilson CI for moving average data 
as the computation is applied a very large number of times.
Efficiency could be improved, but it is not too bad.
"""
from constants import PROJECT_ROOT
from dataprep.coredata import load_coredata, load_additional_data
from dataprep.bloods import filter_bloods
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from matplotlib.dates import YearLocator, DateFormatter
from sklearn.metrics import r2_score


out_path = PROJECT_ROOT / 'results'


# Helpers
def moving_average(df, col, timecol, window='30D', normal_ci=False):
    """Moving average with Gaussian or Wilson CI"""
    df = df.sort_values(by=[timecol])
    df[timecol] = df[timecol].dt.normalize()
    df = df.set_index(timecol)
    df = df[[col]]
    #df = df.resample('D')[col].mean()
    out = df.rolling(window=window, center=True).agg(['mean', 'count'])
    out.columns = [col, 'n']

    if not normal_ci: 
        out['npos'] = out.n * out[col]
        out[['npos', 'n']] = out[['npos', 'n']].astype(int)
        ci = out.apply(lambda x: sm.stats.proportion_confint(nobs=x.n, count=x.npos, method='wilson'), axis=1)
        out['ci'] = ci
        out['ci_low'] = out.ci.apply(lambda x: x[0])
        out['ci_high'] = out.ci.apply(lambda x: x[1])
        out = out.drop(labels=['ci', 'npos'], axis=1)
    else:
        out['se'] = np.sqrt(out[col] * (1 - out[col]) / out.n)
        out['ci_low'] = out[col] - 1.96 * out.se
        out['ci_high'] = out[col] + 1.96 * out.se
        out.loc[out.ci_low < 0, 'ci_low'] = 0
        out.loc[out.ci_high > 1, 'ci_high'] = 1

    out = out.reset_index().drop_duplicates()
    return out


def monthly_average(df, col, timecol, normal_ci=False):
    """Monthly average with Gaussian or Wilson CI"""
    df['yearmonth'] = df[timecol].apply(lambda x: x.strftime('%Y-%m'))
    df.yearmonth = pd.to_datetime(df.yearmonth, format='%Y-%m')
    out = df.groupby('yearmonth')[col].agg(['mean', 'count'])
    out.columns = [col, 'n']
    
    if not normal_ci: 
        out['npos'] = out.n * out[col]
        out[['npos', 'n']] = out[['npos', 'n']].astype(int)
        ci = out.apply(lambda x: sm.stats.proportion_confint(nobs=x.n, count=x.npos, method='wilson'), axis=1)
        out['ci'] = ci
        out['ci_low'] = out.ci.apply(lambda x: x[0])
        out['ci_high'] = out.ci.apply(lambda x: x[1])
        out = out.drop(labels=['ci', 'npos'], axis=1)
    else:
        out['se'] = np.sqrt(out[col] * (1 - out[col]) / out.n)
        out['ci_low'] = out[col] - 1.96 * out.se
        out['ci_high'] = out[col] + 1.96 * out.se
        out.loc[out.ci_low < 0, 'ci_low'] = 0
        out.loc[out.ci_high > 1, 'ci_high'] = 1

    out = out.reset_index()
    out['year'] = out.yearmonth.dt.year
    out['month'] = out.yearmonth.dt.month
    out = out.sort_values(by='yearmonth')
    return out


# ==== 1. Timelines for FIT positivity and clinical symptoms ====

# ---- 1.1. Prepare data
#region

# Paths
data_path = PROJECT_ROOT / 'data_before_inclusion'

# Get data
cdata = load_coredata(data_path)
adata = load_additional_data(data_path)
fit = cdata.fit
fit.shape

fit = fit.loc[fit.gp == 1]
fit.shape

adult_patients = cdata.demo.loc[cdata.demo.age_at_fit >= 18].patient_id
fit = fit.loc[fit.patient_id.isin(adult_patients)]
fit.shape

## Exclude CRC patients when CRC occurred before earliest FIT value
subj_ex = cdata.diagmin.loc[cdata.diagmin.crc_before_fit==1].patient_id.unique()
subj = np.setdiff1d(cdata.fit.patient_id, subj_ex)
fit = fit.loc[fit.patient_id.isin(subj)]
fit.shape

##
days_before_fit_blood = 365
days_after_fit_blood = 14
npat_blood = 30000
adata.bloods = filter_bloods(adata.bloods, cdata.fit, cdata.diagmin, days_before_fit_blood, days_after_fit_blood, npat_blood)
test_codes = ['PLT', 'MCV']
for c in test_codes:
    df = adata.bloods.loc[adata.bloods.test_code == c]
    subj = df.patient_id.unique()
    fit = fit.loc[fit.patient_id.isin(subj)]
print(fit.shape)

fit['fit_cat'] = pd.cut(fit.fit_val, bins=[0,2,10,100,10000], labels=['0-1.9', '2-9.9', '10-99.9', '≥100'], right=False)
fit.fit_cat = fit.fit_cat.astype(str)

fit = fit.loc[fit.fit_date >= "2017-01-01"]

# Exclude march as very little data
fit = fit.loc[fit.fit_date < "2024-03-01"]

# Symptom names
repl = {'abdomass':'Abdominal mass', 
            'abdopain':'Abdominal pain', 
            'anaemia':'Anaemia',
            'bloat':'Bloating', 
            'bloodsympt':'Blood in stool', 
            'bowelhabit':'Change in bowel habit', 
            'constipation':'Constipation', 
            'diarr':'Diarrhoea',
            'fh':'Family history of CRC', 
            'fatigue':'Fatigue',
            'ida':'Iron deficiency anaemia', 
            'inflam':'Inflammation',
            'low_iron':'Low iron',
            'rectalbleed':'Rectal bleeding',
            'rectalpain':'Rectal pain', 
            'rectalulcer':'Rectal ulcer', 
            'rectalmass':'Rectal mass', 
            'tarry':'Melaena', 
            'thrombo':
            'Thrombocytosis', 
            'wl':'Weight loss', 
            }
repl = {'symptom_' + key: val for key, val in repl.items() if key != 'fit10'}
repl['fit10'] =  'FIT ≥ 10 µg/g'

## Fit positivity by symptom category
sym_cols = [c for c in fit.columns if c.startswith('sympt')]
sym_cols.sort()
out = pd.DataFrame([0, 1], columns=['fit10'])
for c in sym_cols:
    o = fit.groupby(c).fit10.mean().rename(c)
    o.index.name = 'fit10'
    o = o.reset_index()
    out = out.merge(o, on='fit10')
out = out.transpose()
out['delta'] = out.iloc[:,1] - out.iloc[:,0]
out = out.round(3) * 100
out
#endregion

## Symptom time series with moving average
sym_cols = [c for c in fit.columns if c.startswith('sympt')]
sym_cols.sort()
cols = ['fit10'] + sym_cols

# ---- 1.2. Plot moving average
#region
for window in ['30D', '60D', '180D']:
    print('\n', window)

    # Keep only symptoms where at least 10 patients have them in at least one time window
    # (use normal_ci here as it is faster and we only care about point estim)
    cols_keep = []
    for i, c in enumerate(cols):
        ma30 = moving_average(fit, c, 'fit_date', window=window, normal_ci=True)
        ma30['nmin'] = ma30.n * ma30[c]
        ma30 = ma30.loc[ma30.nmin >= 10]
        ma30 = ma30.loc[ma30.n >= 10]
        if ma30.shape[0] == 0:
            print('Dropping column', c)
            continue
        else:
            cols_keep.append(c)

    # Plot time series
    xlim = (pd.to_datetime('2016-11-01'), fit.fit_date.max())

    fig, ax = plt.subplots(5, 4, figsize=(15, 8), tight_layout=True)
    ax = ax.flatten()

    for i, c in enumerate(cols_keep):
        print(c)
        ma30 = moving_average(fit, c, 'fit_date', window=window)

        # Ensure at least 10 positive observations in each time window
        ma30['nmin'] = ma30.n * ma30[c]
        ma30 = ma30.loc[ma30.nmin >= 10]
        ma30 = ma30.loc[ma30.n >= 10]

        #label = repl[c] + ', ' + window + ' MA'
        label = repl[c]

        #ax[i].plot(ma30.fit_date.to_numpy(), ma30.ci_low.to_numpy(), linestyle='dashed', color='C0', alpha=0.75)
        #ax[i].plot(ma30.fit_date.to_numpy(), ma30.ci_high.to_numpy(), linestyle='dashed', color='C0', alpha=0.75)
        ax[i].fill_between(ma30.fit_date, y1=ma30.ci_low, y2=ma30.ci_high, facecolor='gray', alpha=0.75)
        ax[i].plot(ma30.fit_date.to_numpy(), ma30[c].to_numpy(), label=label, alpha=1)
        ax[i].set(title=label, xlim=xlim)

        # (from Gemini)
        ax[i].xaxis.set_major_locator(YearLocator())
        ax[i].xaxis.set_major_formatter(DateFormatter("%Y"))

    for i in range(len(cols_keep), 20):
        print(i)
        ax[i].set_visible(False)

    window_label = re.findall('\d+', window)[0]
    plt.savefig(out_path / ('fit_and_symptoms_ma-' + window_label + '.png'), dpi=300, facecolor='white')
    #plt.show()
    plt.close()
#endregion

# ---- 1.3. Plot monthly average
#region

## Symptom time series with monthly average
cols_keep = []
for i, c in enumerate(cols):
    ma30 = monthly_average(fit, c, 'fit_date', normal_ci=True)
    ma30['nmin'] = ma30.n * ma30[c]
    ma30 = ma30.loc[ma30.nmin >= 10]
    ma30 = ma30.loc[ma30.n >= 10]
    if ma30.shape[0] == 0:
        print('Dropping column', c)
        continue
    else:
        cols_keep.append(c)

## Plot time series
xlim = (fit.yearmonth.min(), fit.yearmonth.max())
fig, ax = plt.subplots(5, 4, figsize=(15, 8), tight_layout=True)
ax = ax.flatten()

for i, c in enumerate(cols_keep):
    print(c)
    ma30 = monthly_average(fit, c, 'fit_date')

    # Ensure at least 10 obs in each month
    ma30['nmin'] = ma30.n * ma30[c]
    ma30 = ma30.loc[ma30.nmin >= 10]
    ma30 = ma30.loc[ma30.n >= 10]

    label = repl[c]

    #ax[i].plot(ma30.fit_date.to_numpy(), ma30.ci_low.to_numpy(), linestyle='dashed', color='C0', alpha=0.75)
    #ax[i].plot(ma30.fit_date.to_numpy(), ma30.ci_high.to_numpy(), linestyle='dashed', color='C0', alpha=0.75)
    ax[i].fill_between(ma30.yearmonth, y1=ma30.ci_low, y2=ma30.ci_high, facecolor='gray', alpha=0.75)
    ax[i].plot(ma30.yearmonth.to_numpy(), ma30[c].to_numpy(), label=label, alpha=1)
    ax[i].set(title=label, xlim=xlim)

for i in range(len(cols_keep), 20):
    print(i)
    ax[i].set_visible(False)

plt.savefig(out_path / ('fit_and_symptoms_monthly.png'), dpi=300, facecolor='white')
plt.savefig(out_path / ('fit_and_symptoms_monthly.svg'), dpi=300, facecolor='white')
#plt.show()
plt.close()


#endregion


# ==== 2. Time series of FIT positivity: by buffer, with bloodsympt, with num test ====

# ---- 2.1. Prepare data ----
#region
date_from = fit.loc[fit.stool_pot == 1].fit_date.min()

fit_pre = fit.loc[fit.fit_date < '2021-07-01'].copy()
fit_pre.fit_date.max()

fit_pre['stool_pot'] = 1

fit_post = fit.loc[fit.fit_date >= date_from].copy()
fit_post.fit_date.min()  # Note: after mid april, so fit_middle still has april in it

fit_middle = fit.loc[(fit.fit_date >= '2021-07-01') & (fit.fit_date < date_from)].copy()

assert fit.shape[0] == (fit_pre.shape[0] + fit_middle.shape[0] + fit_post.shape[0])
#endregion

# ---- 2.2. Moving average, by buffer ----
#region
def _filter(ma0, col='fit10'):
    ma0['nmin'] = ma0[col] * ma0.n
    ma0 = ma0.loc[ma0.nmin >= 10]
    print(ma0.nmin.min())
    return ma0


for window in ['30D', '60D', '180D']:

    fig, ax = plt.subplots(figsize=(10, 6))

    ma0 = moving_average(fit_pre, 'fit10', 'fit_date', window=window)
    ma0 = _filter(ma0)
    ax.plot(ma0.fit_date.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Stool pot (mostly)', alpha=1)
    ax.fill_between(ma0.fit_date, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)

    fit2 = fit_post.loc[fit_post.stool_pot == 1].copy()
    ma2 = moving_average(fit2, 'fit10', 'fit_date', window=window)
    ma2 = _filter(ma2)

    ax.plot(ma2.fit_date.to_numpy(), ma2.fit10.to_numpy(), color='C0', label=None, alpha=1)
    ax.fill_between(ma2.fit_date, y1=ma2.ci_low, y2=ma2.ci_high, facecolor='C0', alpha=0.2)

    fit1 = fit_post.loc[fit_post.stool_pot == 0].copy()
    ma1 = moving_average(fit1, 'fit10', 'fit_date', window=window)
    ma1 = _filter(ma1)
    ax.plot(ma1.fit_date.to_numpy(), ma1.fit10.to_numpy(), color='C1', label='Buffer only', alpha=1)
    ax.fill_between(ma1.fit_date, y1=ma1.ci_low, y2=ma1.ci_high, facecolor='C1', alpha=0.2)

    ma3 = moving_average(fit_middle, 'fit10', 'fit_date', window=window)
    ma3 = _filter(ma3)
    ax.plot(ma3.fit_date.to_numpy(), ma3.fit10.to_numpy(), color='C2', label='Mix', alpha=1)
    ax.fill_between(ma3.fit_date, y1=ma3.ci_low, y2=ma3.ci_high, facecolor='C2', alpha=0.2)

    #ax.hlines(y=[0.1, 0.15, 0.2], xmin=fit.fit_date.min(), xmax=fit.fit_date.max(), color='red', alpha=0.5,
    #        linestyle='--')
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
    ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion with FIT ≥ 10 μg/g')
    ax.legend(frameon=False)
    ax.grid(which='major', alpha=0.5)

    window_label = re.findall('\d+', window)[0]

    plt.savefig(out_path / ('fit-positivity-by-buffer_ma-' + window_label + '.png'), dpi=300, facecolor='white')
    plt.close()
#endregion

# ---- 2.3. Monthy average, by buffer ----
#region
fig, ax = plt.subplots(figsize=(10, 6))

ma0 = monthly_average(fit_pre, 'fit10', 'fit_date')
ma0 = _filter(ma0)
ax.plot(ma0.yearmonth.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Stool pot (mostly)', alpha=1)
ax.fill_between(ma0.yearmonth, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)

fit2 = fit_post.loc[fit_post.stool_pot == 1].copy()
ma2 = monthly_average(fit2, 'fit10', 'fit_date')
ma2 = _filter(ma2)
ax.plot(ma2.yearmonth.to_numpy(), ma2.fit10.to_numpy(), color='C0', label=None, alpha=1)
ax.fill_between(ma2.yearmonth, y1=ma2.ci_low, y2=ma2.ci_high, facecolor='C0', alpha=0.2)

fit1 = fit_post.loc[fit_post.stool_pot == 0].copy()
ma1 = monthly_average(fit1, 'fit10', 'fit_date')
ma1 = _filter(ma1)
ax.plot(ma1.yearmonth.to_numpy(), ma1.fit10.to_numpy(), color='C1', label='Buffer only', alpha=1)
ax.fill_between(ma1.yearmonth, y1=ma1.ci_low, y2=ma1.ci_high, facecolor='C1', alpha=0.2)

ma3 = monthly_average(fit_middle, 'fit10', 'fit_date')
ma3 = _filter(ma3)
ax.plot(ma3.yearmonth.to_numpy(), ma3.fit10.to_numpy(), color='C2', label='Mix', alpha=1)
ax.fill_between(ma3.yearmonth, y1=ma3.ci_low, y2=ma3.ci_high, facecolor='C2', alpha=0.2)

#ax.hlines(y=[0.1, 0.15, 0.2], xmin=fit.yearmonth.min(), xmax=fit.yearmonth.max(), color='red', alpha=0.5,
#          linestyle='--')
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion with FIT ≥ 10 μg/g')
ax.legend(frameon=False)
ax.grid(which='major', alpha=0.5)

plt.savefig(out_path / 'fit-positivity-by-buffer_monthly.png', dpi=300, facecolor='white')
plt.savefig(out_path / 'fit-positivity-by-buffer_monthly.svg', dpi=300, facecolor='white')
plt.close()

# Save underlying monthly data
ma0 = monthly_average(fit_pre, 'fit10', 'fit_date')
fit2 = fit_post.loc[fit_post.stool_pot == 1].copy()
ma2 = monthly_average(fit2, 'fit10', 'fit_date')
fit1 = fit_post.loc[fit_post.stool_pot == 0].copy()
ma1 = monthly_average(fit1, 'fit10', 'fit_date')
ma3 = monthly_average(fit_middle, 'fit10', 'fit_date')

ma0['period'] = 'prebuff'
ma1['period'] = 'buffer_only'
ma2['period'] = 'stool_pot_only'
ma3['period'] = 'mix'

data = pd.concat(objs=[ma0, ma1, ma2, ma3], axis=0)
data['n_fit10'] = (data.n * data.fit10).astype(int)
data['p_fit10'] = (data.fit10 * 100).round(2).astype(str) + \
    ' (' + (data.ci_low * 100).round(2).astype(str) + ', ' + (data.ci_high * 100).round(2).astype(str)  +')'

index_cols = ['yearmonth', 'year', 'month', 'period']
value_cols = [c for c in data.columns if c not in index_cols]
data.loc[data.n_fit10 < 10, value_cols] = 'Not available'

first_cols = ['year', 'month', 'n', 'n_fit10', 'fit10']
data = data[first_cols + [c for c in data.columns if c not in first_cols]]

data = data.drop(labels=['yearmonth', 'fit10', 'ci_low', 'ci_high'], axis=1)


periods = ['prebuff', 'mix', 'stool_pot_only', 'buffer_only']
time_label = {'prebuff': 'Mostly stool pot (2017/01 - 2021/06)',
             'mix': 'Mix of stool pot and buffer (2021-07 - 2022-04)',
             'buffer_only': 'Buffer only (2022/05 - 2024/02)',
             'stool_pot_only': 'Stool pot only (2022/05 - 2024/02)'}

print(fit_pre.fit_date.min(), fit_pre.fit_date.max())
print(fit_middle.fit_date.min(), fit_middle.fit_date.max())
print(fit1.fit_date.min(), fit1.fit_date.max())
print(fit2.fit_date.min(), fit2.fit_date.max())

dfnew = pd.DataFrame()
for p in periods:
    if p in data.period.unique():
        dfsub = data.loc[data.period == p]
        row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
        row = pd.DataFrame([row], columns=dfsub.columns)
        dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)

dfnew = dfnew.drop(labels=['period'], axis=1)

dfnew = dfnew.rename(columns={'year': 'Year', 'month': 'Month', 'n': 'Number of FIT tests',
                              'n_fit10': 'Number of positive FIT tests (≥ 10 µg/g)',
                              'p_fit10': 'Percentage of positive FIT tests (≥ 10 µg/g)'})
dfnew.to_csv(out_path / 'fit-positivity-by-buffer_monthly.csv', index=False)
#endregion

# ---- 2.4. Monthy average, FIT pos and bloodsympt ----
#region
fig, ax = plt.subplots(figsize=(10, 6))

ma0 = monthly_average(fit, 'fit10', 'fit_date')
ma0 = _filter(ma0)
ax.plot(ma0.yearmonth.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Proportion with FIT ≥ 10 µg/g', alpha=1)
ax.fill_between(ma0.yearmonth, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)

ma2 = monthly_average(fit, 'symptom_bloodsympt', 'fit_date')
ma2 = _filter(ma2, 'symptom_bloodsympt')
ax.plot(ma2.yearmonth.to_numpy(), ma2.symptom_bloodsympt.to_numpy(), color='C1', label='Proportion with blood in stool', alpha=1)
ax.fill_between(ma2.yearmonth, y1=ma2.ci_low, y2=ma2.ci_high, facecolor='C1', alpha=0.2)

ma3 = monthly_average(fit, 'symptom_rectalbleed', 'fit_date')
ma3 = _filter(ma3, 'symptom_rectalbleed')
ax.plot(ma3.yearmonth.to_numpy(), ma3.symptom_rectalbleed.to_numpy(), color='C2', label='Proportion with rectal bleeding', alpha=1)
ax.fill_between(ma3.yearmonth, y1=ma3.ci_low, y2=ma3.ci_high, facecolor='C2', alpha=0.2)

ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion')
ax.legend(frameon=False)
ax.grid(which='major', alpha=0.5)

plt.savefig(out_path / 'fit-positivity_and_bloodsympt_monthly.png', dpi=300, facecolor='white')
plt.savefig(out_path / 'fit-positivity_and_bloodsympt_monthly.svg', dpi=300, facecolor='white')
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

ma2 = monthly_average(fit, 'symptom_bloodsympt', 'fit_date')
ma2 = _filter(ma2, 'symptom_bloodsympt')
ax.plot(ma2.yearmonth.to_numpy(), ma2.symptom_bloodsympt.to_numpy(), color='C1', label='Proportion with blood in stool', alpha=1)
ax.fill_between(ma2.yearmonth, y1=ma2.ci_low, y2=ma2.ci_high, facecolor='C1', alpha=0.2)

ma3 = monthly_average(fit, 'symptom_rectalbleed', 'fit_date')
ma3 = _filter(ma3, 'symptom_rectalbleed')
ax.plot(ma3.yearmonth.to_numpy(), ma3.symptom_rectalbleed.to_numpy(), color='C2', label='Proportion with rectal bleeding', alpha=1)
ax.fill_between(ma3.yearmonth, y1=ma3.ci_low, y2=ma3.ci_high, facecolor='C2', alpha=0.2)

ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion')
ax.legend(frameon=False)
ax.grid(which='major', alpha=0.5)

plt.savefig(out_path / 'bloodsympt_monthly.png', dpi=300, facecolor='white')
plt.savefig(out_path / 'bloodsympt_monthly.svg', dpi=300, facecolor='white')
plt.close()
#endregion

# ---- 2.5. Moving average, FIT pos and bloodsympt ----
#region
for window in ['30D', '60D', '180D']:
    fig, ax = plt.subplots(figsize=(10, 6))

    ma0 = moving_average(fit, 'fit10', 'fit_date', window)
    ma0 = _filter(ma0)
    ax.plot(ma0.fit_date.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Proportion with FIT ≥ 10 µg/g', alpha=1)
    ax.fill_between(ma0.fit_date, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)

    ma2 = moving_average(fit, 'symptom_bloodsympt', 'fit_date', window)
    ma2 = _filter(ma2, 'symptom_bloodsympt')
    ax.plot(ma2.fit_date.to_numpy(), ma2.symptom_bloodsympt.to_numpy(), color='C1', label='Proportion with blood in stool', alpha=1)
    ax.fill_between(ma2.fit_date, y1=ma2.ci_low, y2=ma2.ci_high, facecolor='C1', alpha=0.2)

    ma3 = moving_average(fit, 'symptom_rectalbleed', 'fit_date', window)
    ma3 = _filter(ma3, 'symptom_rectalbleed')
    ax.plot(ma3.fit_date.to_numpy(), ma3.symptom_rectalbleed.to_numpy(), color='C2', label='Proportion with rectal bleeding', alpha=1)
    ax.fill_between(ma3.fit_date, y1=ma3.ci_low, y2=ma3.ci_high, facecolor='C2', alpha=0.2)

    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
    ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion')
    ax.legend(frameon=False)
    ax.grid(which='major', alpha=0.5)

    window_label = re.findall('\d+', window)[0]

    plt.savefig(out_path / ('fit-positivity_and_bloodsympt_ma-' + window_label + '.png'), dpi=300, facecolor='white')
    plt.close()
#endregion

# ---- 2.6. Monthy FIT pos and number of tests ----
#region
fig, ax = plt.subplots(figsize=(10, 6))

ma0 = monthly_average(fit, 'fit10', 'fit_date')
ma0 = _filter(ma0)
ax.plot(ma0.yearmonth.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Monthly proportion with FIT ≥ 10 µg/g', alpha=1)
ax.fill_between(ma0.yearmonth, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion with FIT ≥ 10 µg/g')

ax2 = ax.twinx()
ax2.plot(ma0.yearmonth.to_numpy(), ma0.n.to_numpy(), color='C1', label='Monthly number of tests', alpha=1)
ax2.set(ylim=(0, ma0.n.max() + 100), ylabel='Number of tests')

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2

ax.legend(handles, labels, frameon=False, loc='upper left')
ax.grid(which='major', alpha=0.5)

plt.savefig(out_path / 'fit-positivity_and_numtest_monthly.png', dpi=300, facecolor='white')
plt.savefig(out_path / 'fit-positivity_and_numtest_monthly.svg', dpi=300, facecolor='white')
plt.close()

## Get underlying data table
data = monthly_average(fit, 'fit10', 'fit_date')
data['n_fit10'] = (data.n * data.fit10).astype(int)
data['p_fit10'] = (data.fit10 * 100).round(2).astype(str) + \
    ' (' + (data.ci_low * 100).round(2).astype(str) + ', ' + (data.ci_high * 100).round(2).astype(str)  +')'

index_cols = ['yearmonth', 'year', 'month']
value_cols = [c for c in data.columns if c not in index_cols]
data.loc[data.n_fit10 < 10, value_cols] = 'Not available'

first_cols = ['year', 'month', 'n', 'n_fit10', 'fit10']
data = data[first_cols + [c for c in data.columns if c not in first_cols]]
data = data.drop(labels=['yearmonth', 'fit10', 'ci_low', 'ci_high'], axis=1)

data = data.rename(columns={'year': 'Year', 'month': 'Month', 'n': 'Number of FIT tests',
                              'n_fit10': 'Number of positive FIT tests (≥ 10 µg/g)',
                              'p_fit10': 'Percentage of positive FIT tests (≥ 10 µg/g)'})
data.to_csv(out_path / 'fit-positivity_monthly.csv', index=False)


## Explore underlying table
data = monthly_average(fit, 'fit10', 'fit_date')
data['n_fit10'] = (data.n * data.fit10).astype(int)
data['p_fit10'] = (data.fit10 * 100).round(2)
data.loc[data.n_fit10 < 10]

data.loc[data.yearmonth < "2021-09-01"].p_fit10.describe(percentiles=[0.025, 0.01, 0.05, 0.1, 0.9, 0.95, 0.975, 0.99])
data.loc[data.yearmonth >= "2021-09-01"]

data.loc[data.yearmonth < "2021-07-01"].n.describe(percentiles=[0.025, 0.01, 0.05, 0.1, 0.9, 0.95, 0.975, 0.99])
data.loc[data.yearmonth >= "2021-07-01"]

#endregion

# ---- 2.7. Moving average: FIT pos and number of tests ----
#region

for window in ['30D', '60D', '180D']:
    fig, ax = plt.subplots(figsize=(10, 6))

    ma0 = moving_average(fit, 'fit10', 'fit_date', window)
    ma0 = _filter(ma0)
    ax.plot(ma0.fit_date.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Monthly proportion with FIT ≥ 10 µg/g', alpha=1)
    ax.fill_between(ma0.fit_date, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
    ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion with FIT ≥ 10 µg/g')

    ax2 = ax.twinx()
    ax2.plot(ma0.fit_date.to_numpy(), ma0.n.to_numpy(), color='C1', label='Monthly number of tests', alpha=1)
    ax2.set(ylim=(0, ma0.n.max() + 100), ylabel='Number of tests')

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    ax.legend(handles, labels, frameon=False, loc='upper left')
    ax.grid(which='major', alpha=0.5)

    window_label = re.findall('\d+', window)[0]
    plt.savefig(out_path / ('fit-positivity_and_numtest_ma-' + window_label + '.png'), dpi=300, facecolor='white')
    plt.close()

#endregion

# ---- 2.8. Correlation: FIT pos and number of tests ----
# code partly from Gemini
#region
ma0 = monthly_average(fit, 'fit10', 'fit_date')
ma0['nmin'] = ma0['fit10'] * ma0.n
ma0.sort_values(by='nmin', ascending=False)
ma0 = ma0.loc[ma0.nmin >= 10]
ma0.shape[0]

# Plot monthly number of tests aganst monthly FIT positivity
#fig, ax = plt.subplots(1, 2, figsize=(14, 7))
fig, ax = plt.subplots(1, 3, figsize=(21, 7))

## .... Before July 2020
msub = ma0.loc[ma0.yearmonth < '2020-07-01']
x = msub['n']
y = msub['fit10']

## Fit a linear regression model with statsmodels.OLS
X = sm.add_constant(x)  # Add constant term for intercept
model = sm.OLS(y, X).fit()  # Fit the model

# Get slope, intercept, and p-value
#m, b = np.polyfit(x, y, 1)  # Linear regression (slope, intercept)
m = model.params[1]  # Slope coefficient
b = model.params[0]  # Intercept
p_value = model.pvalues[1]  # P-value for slope coefficient (index 1)

# Generate x values for the regression line
x_fit = np.linspace(min(x), max(x), 100)

#y_pred = m * x + b
y_pred = model.predict(X) 
y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
if p_value < 0.001:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ < 0.001)"
else:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ = " + f"{p_value:.3f})"

ax[0].scatter(x, y)
ax[0].plot(x_fit, m * x_fit + b, color='red', label=label)
ax[0].set(xlabel='Monthly number of FIT tests', ylabel='Monthly proportion with FIT ≥ 10 µg/g',
       title='2017 Jan - 2020 Jun')
ax[0].legend(frameon=False)

## .... From July 2020 (when num tests starts to increase)
msub = ma0.loc[ma0.yearmonth >= '2020-07-01']
x = msub['n']
y = msub['fit10']

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
m = model.params[1]
b = model.params[0]
p_value = model.pvalues[1]

x_fit = np.linspace(min(x), max(x), 100)

#y_pred = m * x + b
y_pred = model.predict(X) 
y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
label = f"Linear Regression (R² = {r2:.2f})"

y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
if p_value < 0.001:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ < 0.001)"
else:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ = " + f"{p_value:.3f})"

ax[1].scatter(x, y)
ax[1].plot(x_fit, m * x_fit + b, color='red', label=label)
ax[1].set(xlabel='Monthly number of FIT tests', ylabel='Monthly proportion with FIT ≥ 10 µg/g',
       title='2020 Jul - 2024 Feb')
ax[1].legend(frameon=False)


## .... From September 2021 (when FIT pos starts to increase)
msub = ma0.loc[ma0.yearmonth >= '2021-09-01']
x = msub['n']
y = msub['fit10']

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
m = model.params[1]
b = model.params[0]
p_value = model.pvalues[1]

x_fit = np.linspace(min(x), max(x), 100)

#y_pred = m * x + b
y_pred = model.predict(X) 
y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
label = f"Linear Regression (R² = {r2:.2f})"

y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
if p_value < 0.001:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ < 0.001)"
else:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ = " + f"{p_value:.3f})"

ax[2].scatter(x, y)
ax[2].plot(x_fit, m * x_fit + b, color='red', label=label)
ax[2].set(xlabel='Monthly number of FIT tests', ylabel='Monthly proportion with FIT ≥ 10 µg/g',
       title='2021 Sep - 2024 Feb')
ax[2].legend(frameon=False)

plt.savefig(out_path / 'fit-positivity_and_numtest_corr.png', dpi=300, facecolor='white')
plt.savefig(out_path / 'fit-positivity_and_numtest_corr.svg', dpi=300, facecolor='white')
plt.close()



#----------- same graph as above but only for periods when corr is present

# Plot monthly number of tests aganst monthly FIT positivity
#fig, ax = plt.subplots(1, 2, figsize=(14, 7))
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

## .... From July 2020 (when num tests starts to increase)
msub = ma0.loc[ma0.yearmonth >= '2020-07-01']
x = msub['n']
y = msub['fit10']

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
m = model.params[1]
b = model.params[0]
p_value = model.pvalues[1]

x_fit = np.linspace(min(x), max(x), 100)

#y_pred = m * x + b
y_pred = model.predict(X) 
y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
label = f"Linear Regression (R² = {r2:.2f})"

y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
if p_value < 0.001:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ < 0.001)"
else:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ = " + f"{p_value:.3f})"

ax[0].scatter(x, y)
ax[0].plot(x_fit, m * x_fit + b, color='red', label=label)
ax[0].set(xlabel='Monthly number of FIT tests', ylabel='Monthly proportion with FIT ≥ 10 µg/g',
       title='2020 Jul - 2024 Feb')
ax[0].legend(frameon=False)


## .... From September 2021 (when FIT pos starts to increase)
msub = ma0.loc[ma0.yearmonth >= '2021-09-01']
x = msub['n']
y = msub['fit10']

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
m = model.params[1]
b = model.params[0]
p_value = model.pvalues[1]

x_fit = np.linspace(min(x), max(x), 100)

#y_pred = m * x + b
y_pred = model.predict(X) 
y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
label = f"Linear Regression (R² = {r2:.2f})"

y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
if p_value < 0.001:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ < 0.001)"
else:
    label = f"Linear Regression (R² = {r2:.2f}" + ", $p_{slope}$ = " + f"{p_value:.3f})"

ax[1].scatter(x, y)
ax[1].plot(x_fit, m * x_fit + b, color='red', label=label)
ax[1].set(xlabel='Monthly number of FIT tests', ylabel='Monthly proportion with FIT ≥ 10 µg/g',
       title='2021 Sep - 2024 Feb')
ax[1].legend(frameon=False)

plt.savefig(out_path / 'fit-positivity_and_numtest_corr_positive.png', dpi=300, facecolor='white')
plt.savefig(out_path / 'fit-positivity_and_numtest_corr_positive.svg', dpi=300, facecolor='white')
plt.close()


#endregion

# ---- 2.9. Monthy FIT pos and stool_pot ----
#region
fig, ax = plt.subplots(figsize=(10, 6))

date_thr = fit.loc[fit.stool_pot == 1].fit_date.min()
print(date_thr)
fitsub = fit.loc[fit.fit_date >= "2022-05-01"]

ma0 = monthly_average(fitsub, 'stool_pot', 'fit_date')
ma0 = _filter(ma0, col = 'stool_pot')
ax.plot(ma0.yearmonth.to_numpy(), ma0.stool_pot.to_numpy(), color='C0', label='Monthly proportion in stool pot', alpha=1)
ax.fill_between(ma0.yearmonth, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.40])
ax.set(ylim=(0, 0.4), xlabel='Date', ylabel='Proportion of tests in stool pot')
ax.legend(frameon=False, loc='upper left')
ax.grid(which='major', alpha=0.5)

plt.savefig(out_path / 'stool_pot.png', dpi=300, facecolor='white')
plt.close()

## Get underlying data table
data = monthly_average(fitsub.copy(), 'stool_pot', 'fit_date')
data['n_stool'] = (data.n * data.stool_pot).astype(int)
data['p_stool'] = (data.stool_pot * 100).round(2).astype(str) + \
    ' (' + (data.ci_low * 100).round(2).astype(str) + ', ' + (data.ci_high * 100).round(2).astype(str)  +')'

index_cols = ['yearmonth', 'year', 'month']
value_cols = [c for c in data.columns if c not in index_cols]
data.loc[data.n_stool < 10, value_cols] = 'Not available'

first_cols = ['year', 'month', 'n', 'n_stool', 'p_stool']
data = data[first_cols + [c for c in data.columns if c not in first_cols]]
data = data.drop(labels=['yearmonth', 'stool_pot', 'ci_low', 'ci_high'], axis=1)

data = data.rename(columns={'year': 'Year', 'month': 'Month', 'n': 'Number of FIT tests',
                              'n_stool': 'Number of FIT tests in stool pot',
                              'p_stool': 'Percentage of FIT tests in stool pot'})
data.to_csv(out_path / 'stool_pot.csv', index=False)
#endregion

# ---- 2.10. Estimate increase in FIT positivity ----
#region

date_from = fit.loc[fit.stool_pot == 1].fit_date.min()
fit_pre = fit.loc[fit.fit_date < '2021-07-01'].copy()
fit_pre['stool_pot'] = 1
fit_post = fit.loc[fit.fit_date >= date_from].copy()



ma0 = monthly_average(fit_pre, 'fit10', 'fit_date')
ma0 = _filter(ma0)

fit1 = fit_post.loc[fit_post.stool_pot == 0].copy()
ma1 = monthly_average(fit1, 'fit10', 'fit_date')
ma1 = _filter(ma1)




ma0['buffer'] = 0
ma1['buffer'] = 1
ma = pd.concat(objs=[ma0, ma1], axis=0)
ma = ma.loc[ma.year >= 2020]
years = np.arange(ma.year.min(), ma.year.max() + 1, 1)
months = np.arange(1, 13, 1)
yearmonths = []
for y in years:
    for m in months:
        yearmonths.append(str(y) + '-' + str(m).zfill(2) + '-01')
df = pd.DataFrame(yearmonths, columns=['yearmonth'])
df.yearmonth = pd.to_datetime(df.yearmonth)
ma = ma.merge(df, how='outer', on='yearmonth')
ma = ma.sort_values(by='yearmonth')
ma = ma.loc[ma.yearmonth < '2024-03-01']
ma['x'] = np.arange(ma.shape[0])

ma_buf = ma.loc[ma.buffer == 1]
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(ma_buf.x.to_numpy().reshape(-1, 1), ma_buf.fit10.to_numpy().reshape(-1, 1))

ma['pred_buf'] = clf.predict(ma.x.to_numpy().reshape(-1, 1))

ma_prebuf = ma.loc[ma.buffer == 0]
clf = LinearRegression()
clf.fit(ma_prebuf.x.to_numpy().reshape(-1, 1), ma_prebuf.fit10.to_numpy().reshape(-1, 1))
ma['pred_prebuf'] = clf.predict(ma.x.to_numpy().reshape(-1, 1))

ma_pred_pre = ma.loc[ma.buffer == 0]
ma_pred_buf = ma.loc[ma.yearmonth >= '2021-06-01']



fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ma0.yearmonth.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Stool pot (mostly)', alpha=0.5)
ax.fill_between(ma0.yearmonth, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)

ax.plot(ma1.yearmonth.to_numpy(), ma1.fit10.to_numpy(), color='C1', label='Buffer only', alpha=0.5)
ax.fill_between(ma1.yearmonth, y1=ma1.ci_low, y2=ma1.ci_high, facecolor='C1', alpha=0.2)

ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion with FIT ≥ 10 μg/g')
ax.legend(frameon=False)
ax.grid(which='major', alpha=0.5)

ax.plot(ma_pred_pre.yearmonth.to_numpy(), ma_pred_pre.pred_prebuf.to_numpy(), color='red', linestyle='dashed')
ax.plot(ma_pred_buf.yearmonth.to_numpy(), ma_pred_buf.pred_buf.to_numpy(), color='red', linestyle='dashed')

ma['delta'] = ma.pred_buf - ma.pred_prebuf
delta = ma.loc[ma.yearmonth == '2021-06-01', 'delta'].item()
delta = np.round(delta, 3)
ax.annotate('Difference: ' + str(delta), xy=(pd.Timestamp('2021-04-01'), 0.105), color='red')


plt.savefig(out_path / 'fit-positivity_buffer-vs-stool.png', dpi=300, facecolor='white')
plt.savefig(out_path / 'fit-positivity_buffer-vs-stool.svg', dpi=300, facecolor='white')
plt.close()

#endregion

# ---- 2.11. Overall panel figure for population change ----
#region

fig = plt.figure(layout='constrained', figsize=(12, 8))
subfigs = fig.subfigures(2, 2, wspace=0.1, hspace=0.1)
subfigs = subfigs.flatten()

titles = ['A. Monthly FIT positivity and testing volume over time',
          'B. Monthly FIT positivity by buffer device',
          'C. Monthly FIT positivity and blood in stool over time',
          'D. Correlation of monthly FIT positivity and number of tests']

for i, title in enumerate(titles):
    subfigs[i].suptitle(title, x=0.01, ha='left')


# .... Panel A .... 
#fig, ax = plt.subplots(figsize=(10, 6))
ax = subfigs[0].subplots(1, 1)

ma0 = monthly_average(fit, 'fit10', 'fit_date')
ma0 = _filter(ma0)
ax.plot(ma0.yearmonth.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Monthly proportion with FIT ≥ 10 µg/g', alpha=1)
ax.fill_between(ma0.yearmonth, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion with FIT ≥ 10 µg/g')

ax2 = ax.twinx()
ax2.plot(ma0.yearmonth.to_numpy(), ma0.n.to_numpy(), color='C1', label='Monthly number of tests', alpha=1)
ax2.set(ylim=(0, ma0.n.max() + 100), ylabel='Number of tests')

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2

ax.legend(handles, labels, frameon=False, loc='upper left')
ax.grid(which='major', alpha=0.5)


# .... Panel B .... 
#fig, ax = plt.subplots(figsize=(10, 6))
ax = subfigs[1].subplots(1, 1)

ma0 = monthly_average(fit_pre, 'fit10', 'fit_date')
ma0 = _filter(ma0)
ax.plot(ma0.yearmonth.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Stool pot (mostly)', alpha=1)
ax.fill_between(ma0.yearmonth, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)

fit2 = fit_post.loc[fit_post.stool_pot == 1].copy()
ma2 = monthly_average(fit2, 'fit10', 'fit_date')
ma2 = _filter(ma2)
ax.plot(ma2.yearmonth.to_numpy(), ma2.fit10.to_numpy(), color='C0', label=None, alpha=1)
ax.fill_between(ma2.yearmonth, y1=ma2.ci_low, y2=ma2.ci_high, facecolor='C0', alpha=0.2)

fit1 = fit_post.loc[fit_post.stool_pot == 0].copy()
ma1 = monthly_average(fit1, 'fit10', 'fit_date')
ma1 = _filter(ma1)
ax.plot(ma1.yearmonth.to_numpy(), ma1.fit10.to_numpy(), color='C1', label='Buffer only', alpha=1)
ax.fill_between(ma1.yearmonth, y1=ma1.ci_low, y2=ma1.ci_high, facecolor='C1', alpha=0.2)

ma3 = monthly_average(fit_middle, 'fit10', 'fit_date')
ma3 = _filter(ma3)
ax.plot(ma3.yearmonth.to_numpy(), ma3.fit10.to_numpy(), color='C2', label='Mix', alpha=1)
ax.fill_between(ma3.yearmonth, y1=ma3.ci_low, y2=ma3.ci_high, facecolor='C2', alpha=0.2)

ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion with FIT ≥ 10 μg/g')
ax.legend(frameon=False)
ax.grid(which='major', alpha=0.5)


# .... Panel C ....
#fig, ax = plt.subplots(figsize=(10, 6))
ax = subfigs[2].subplots(1, 1)

ma0 = monthly_average(fit, 'fit10', 'fit_date')
ma0 = _filter(ma0)
ax.plot(ma0.yearmonth.to_numpy(), ma0.fit10.to_numpy(), color='C0', label='Proportion with FIT ≥ 10 µg/g', alpha=1)
ax.fill_between(ma0.yearmonth, y1=ma0.ci_low, y2=ma0.ci_high, facecolor='C0', alpha=0.2)

ma2 = monthly_average(fit, 'symptom_bloodsympt', 'fit_date')
ma2 = _filter(ma2, 'symptom_bloodsympt')
ax.plot(ma2.yearmonth.to_numpy(), ma2.symptom_bloodsympt.to_numpy(), color='C1', label='Proportion with blood in stool', alpha=1)
ax.fill_between(ma2.yearmonth, y1=ma2.ci_low, y2=ma2.ci_high, facecolor='C1', alpha=0.2)

ma3 = monthly_average(fit, 'symptom_rectalbleed', 'fit_date')
ma3 = _filter(ma3, 'symptom_rectalbleed')
ax.plot(ma3.yearmonth.to_numpy(), ma3.symptom_rectalbleed.to_numpy(), color='C2', label='Proportion with rectal bleeding', alpha=1)
ax.fill_between(ma3.yearmonth, y1=ma3.ci_low, y2=ma3.ci_high, facecolor='C2', alpha=0.2)

ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30])
ax.set(ylim=(0, 0.3), xlabel='Date', ylabel='Proportion')
ax.legend(frameon=False)
ax.grid(which='major', alpha=0.5)


# .... Panel D ....
#fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax = subfigs[3].subplots(1, 2)

ma0 = monthly_average(fit, 'fit10', 'fit_date')
ma0['nmin'] = ma0['fit10'] * ma0.n
ma0.sort_values(by='nmin', ascending=False)
ma0 = ma0.loc[ma0.nmin >= 10]
ma0.shape[0]

## .... From July 2020 (when num tests starts to increase)
msub = ma0.loc[ma0.yearmonth >= '2020-07-01']
x = msub['n']
y = msub['fit10']

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
m = model.params[1]
b = model.params[0]
p_value = model.pvalues[1]

x_fit = np.linspace(min(x), max(x), 100)
y_pred = model.predict(X) 
y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
label = f"R² = {r2:.2f}"

y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
if p_value < 0.001:
    label = f"R² = {r2:.2f}" + "\n$p_{slope}$ < 0.001"
else:
    label = f"R² = {r2:.2f}" + "\n$p_{slope}$ = " + f"{p_value:.3f}"

ax[0].scatter(x, y, alpha=0.75)
ax[0].plot(x_fit, m * x_fit + b, color='red', label=label)
ax[0].set(xlabel='Monthly number of FIT tests', ylabel='Monthly proportion with FIT ≥ 10 µg/g',
       title='2020 Jul - 2024 Feb')
ax[0].legend(frameon=False, prop={'size': 10})


## .... From September 2021 (when FIT pos starts to increase)
msub = ma0.loc[ma0.yearmonth >= '2021-09-01']
x = msub['n']
y = msub['fit10']

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
m = model.params[1]
b = model.params[0]
p_value = model.pvalues[1]

x_fit = np.linspace(min(x), max(x), 100)
y_pred = model.predict(X) 
y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
label = f"R² = {r2:.2f}"

y_mean = np.mean(y)
r2 = r2_score(y, y_pred)
if p_value < 0.001:
    label = f"R² = {r2:.2f}" + "\n$p_{slope}$ < 0.001"
else:
    label = f"R² = {r2:.2f}" + "\n$p_{slope}$ = " + f"{p_value:.3f}"

ax[1].scatter(x, y, alpha=0.75)
ax[1].plot(x_fit, m * x_fit + b, color='red', label=label)
ax[1].set(xlabel='Monthly number of FIT tests', ylabel='Monthly proportion with FIT ≥ 10 µg/g',
       title='2021 Sep - 2024 Feb')
ax[1].legend(frameon=False, prop={'size': 10})


plt.savefig(out_path / 'population_change.png', dpi=300, facecolor='white')
plt.savefig(out_path / 'population_change.svg', dpi=300, facecolor='white')
plt.close()

#endregion

