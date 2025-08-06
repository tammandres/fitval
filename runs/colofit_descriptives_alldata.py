"""Compute some descriptives on ALL data, not separated by CRC, for the paper"""
import pandas as pd
from pathlib import Path
from fitval.models import get_model
from fitval.metrics import metric_at_fit_sens

fu = 180

data_path = Path('C:\\Users\\5lJC\Desktop\\dataprep_fitml_and_fitval\\data-colofit_fu-' + str(fu))
df = pd.read_csv(data_path / 'data_matrix.csv')

df[['age_at_fit', 'fit_val', 'blood_PLT', 'blood_MCV', 'blood_HGB', 'ind_gender_M']].describe()

(df.fit_val >= 10).mean() * 100

demo = pd.read_csv(data_path / 'demo.csv')
demo = demo[['patient_id', 'ethnicity']].drop_duplicates()
print(demo.shape, demo.patient_id.nunique())
demo.ethnicity.value_counts(normalize=True) * 100


ethnic_dict = {
    'British':'White',
    'Irish':'White',
    'Any other White background':'White',
    'White and Black Caribbean':'Mixed',
    'White and Black African':'Mixed',
    'White and Asian':'Mixed',
    'Any other mixed background':'Mixed',
    'Indian':'Asian', # or Asian British',
    'Pakistani':'Asian', # or Asian British',
    'Bangladeshi':'Asian', # or Asian British',
    'Any other Asian background':'Asian', # or Asian British',
    'Caribbean':'Black', # or Black British',
    'African':'Black', # or Black British',
    'Any other Black background':'Black', # or Black British',
    'Chinese':'Other Ethnic Groups',
    'Any other ethnic group':'Other Ethnic Groups'}

demo.ethnicity = demo.ethnicity.replace(ethnic_dict)
demo.ethnicity.value_counts(normalize=True) * 100
demo.ethnicity.value_counts(normalize=False)
demo.ethnicity.value_counts(normalize=True).round(3) * 100

# Get diagnosis data
diag = pd.read_csv(data_path / 'diag.csv')
diag.diagnosis_source.value_counts(normalize=True)
diag.diagnosis_date = pd.to_datetime(diag.diagnosis_date)

# Get earliest diagnosis date of each type
e = diag.groupby(['patient_id', 'diagnosis_source']).diagnosis_date.min().reset_index()
e = e.pivot(index='patient_id', columns='diagnosis_source', values='diagnosis_date').reset_index()
e['delta_inpat_path'] = (e.inpat - e.pathology).dt.days
e['delta_inpat_outpat'] = (e.inpat - e.outpat).dt.days

## Percent cases where pathology was at least within 180 days of inpat, or there was pathology but no inpat or outpat
path = e.loc[~e.pathology.isna()]
path.delta_inpat_path.isna().mean()
path.delta_inpat_path.describe(percentiles=[0.01, 0.05, 0.95, 0.99])

path = path.loc[(path.delta_inpat_path >= -180) | (path.inpat.isna() & path.outpat.isna())]
e.patient_id.isin(path.patient_id).mean()

## Percent cases where inpat was main source 
inpat = e.loc[~e.inpat.isna()]
inpat = inpat.loc[~inpat.patient_id.isin(path.patient_id)]
inpat.delta_inpat_outpat.isna().mean()
inpat.delta_inpat_outpat.describe(percentiles=[0.01, 0.05, 0.95, 0.99])
inpat = inpat.loc[(inpat.delta_inpat_outpat >= -180) | (inpat.outpat.isna())]
e.patient_id.isin(inpat.patient_id).mean()

## Percent cases where outpat was main source 
mask = ~(e.patient_id.isin(inpat.patient_id) | e.patient_id.isin(path.patient_id))
mask.mean() * 100


## --- In addtion, check if FIT positives do not change with 365-day fu in some time periods
model = get_model('nottingham-cox')
matrix = {}
for fu in [180, 365]:
    data_path = Path('C:\\Users\\5lJC\Desktop\\dataprep_fitml_and_fitval\\data-colofit_fu-' + str(fu))

    df = pd.read_csv(data_path / 'data_matrix.csv')
    fit = pd.read_csv(data_path / 'fit.csv')
    fit.fit_date = pd.to_datetime(fit.fit_date)

    precovid = fit.loc[fit.fit_date < '2020-03-01']
    covid = fit.loc[(fit.fit_date >= '2020-03-01') & (fit.fit_date < '2021-05-01')]
    post1 = fit.loc[(fit.fit_date >= '2021-05-01') & (fit.fit_date < '2022-01-01')]
    post2 = fit.loc[(fit.fit_date >= '2022-01-01') & (fit.fit_date < '2022-07-01')]
    post3 = fit.loc[(fit.fit_date >= '2022-07-01') & (fit.fit_date < '2023-01-01')]

    fit_data = {'precovid': precovid, 
                'covid': covid, 
                'post1': post1,
                'post2': post2, 
                'post3': post3,
                }

    m = {}
    for label, f in fit_data.items():
        dfsub = df.loc[df.patient_id.isin(f.patient_id)]
        dfsub['y_pred'] = model(dfsub)
        m[label] = dfsub
        print(label, dfsub.shape[0], dfsub.crc.sum())
    matrix[fu] = m


# Note:
#
#  In the time periods that have the same patients with 365 and 180-day follow-ups,
#  the model predictions are the same.
#
#  The patients that test positive for FIT are also the same, as the patients are same.
#  Therefore, reduction in referrals can be the same only if the estimated model thr is the same
#
#  The estimated model threshold may or may not be the same -- it needs to capture the same number of cancers as FIT.
#
for period in ['precovid', 'covid', 'post1', 'post2', 'post3']:
    print('\n---')

    df0 = matrix[180][period].reset_index(drop=True)
    df1 = matrix[365][period].reset_index(drop=True)

    assert (df0.patient_id == df1.patient_id).all()
    test = (df0.fit_val >= 10) == (df1.fit_val >= 10)
    #print(period, test.mean())

    test3 = df0.y_pred == df1.y_pred
    #print(period, test3.mean())

    df0pos = df0.loc[df0.fit_val >= 10]
    df1pos = df1.loc[df1.fit_val >= 10]
    test2 = df0pos.crc == df1pos.crc
    print(period, (~test2).sum(), 'extra fit positive cancers')

    sens0 = df0.loc[df0.crc == 1].patient_id.isin(df0pos.patient_id).mean() * 100
    sens1 = df1.loc[df1.crc == 1].patient_id.isin(df1pos.patient_id).mean() * 100

    print(period, sens0, sens1, 'are the sensitivities with the two follow-ups')

    y_true, y_pred, fit_val = df0.crc.to_numpy(), df0.y_pred.to_numpy(), df0.fit_val.to_numpy()
    m, g = metric_at_fit_sens(y_true, y_pred, fit_val, thr_fit=[10])
    thr = m.loc[m.model == 'model', 'thr'].item()
    print(g.sens_mod.item(), g.sens_fit.item(), g.proportion_reduction_tests.item(), thr)

    y_true, y_pred, fit_val = df1.crc.to_numpy(), df1.y_pred.to_numpy(), df1.fit_val.to_numpy()
    m, g = metric_at_fit_sens(y_true, y_pred, fit_val, thr_fit=[10])
    thr = m.loc[m.model == 'model', 'thr'].item()
    print(g.sens_mod.item(), g.sens_fit.item(), g.proportion_reduction_tests.item(), thr)





