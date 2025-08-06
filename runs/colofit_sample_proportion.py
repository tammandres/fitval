import statsmodels.api as stat
import pandas as pd
import scipy as sci
import matplotlib.pyplot as plt
import numpy as np

# Helper function for Chi-squared test for a difference in proportion
def xtest(n0_pos, n1_pos, n0 = 50818, n1 = 659, return_data=False):
    """"
    Args: n0 - num patients without cancer
          n1 - num patients with cancer
          n0_pos - num patients with outcome (e.g. male gender) among those without cancer
          n1_pos - num patients with outcome (e.g. male gender) among those with cancer
    """
    p0 = n0_pos / n0 * 100
    p1 = n1_pos / n1 * 100
    xstat, p, (obs, exp) = stat.stats.proportions_chisquare(count = [n1_pos, n0_pos], nobs = [n1, n0])
    print("perc_0 {:.1f}, perc_1 {:.1f}, X {:.1f}, p {}".format(p0, p1, xstat, p))

    if return_data:
        return p0, p1, xstat, p, obs, exp


def xtest_loop(df, var):
    values = df[var].drop_duplicates().sort_values()
    for val in values:
        df['ind_v'] = df[var] == val
        s = df.groupby('crc').ind_v.sum()
        n0_pos = s.loc[0]
        n1_pos = s.loc[1]
        print('\n----', val, n0_pos, n1_pos)
        if n1_pos >= 10:
            p0, p1, xstat, p, obs, exp = xtest(n0_pos, n1_pos, return_data=True)
            print(xstat)
            if p < 0.001:
                print('p < 0.001')
            else:
                print(np.round(p, 3))


def mann_whitney_test(df, var):
    v0 = df.loc[df.crc == 0, var]
    v1 = df.loc[df.crc == 1, var]
    out = sci.stats.mannwhitneyu(v0, v1, use_continuity=True, alternative='two-sided')
    print(out.statistic, out.pvalue)
    if out.pvalue < 0.001:
        print('p < 0.001')



# Get overall patient counts
df = pd.read_csv(r'C:\Users\5lJC\Desktop\dataprep_fitml_and_fitval\data-colofit_fu-180\data_matrix.csv')
df.crc.value_counts()
n0 = 50818
n1 = 659

# Gender: chi-square test
#  20972 - males without cancer
#  371 - males with cancer
df['fit10'] = df.fit_val >= 10
df.groupby(['crc']).ind_gender_M.sum()
df.groupby('crc').size()
p0, p1, xstat, p, obs, exp = xtest(20972, 371, return_data=True)
1 - sci.stats.chi2.cdf(xstat, 1)
p

## The result for female gender should mirror that of males
df['ind_gender_notM'] = 1 - df.ind_gender_M
df.groupby(['crc']).ind_gender_notM.sum()
p0, p1, xstat, p, obs, exp = xtest(29846, 288, return_data=True)
1 - sci.stats.chi2.cdf(xstat, 1)
p


# FIT >= 10: chi-square test
#  5617 - fit positives without cancer (n0_pos)
#  577 - fit positives with cancer (n1_pos)
#  n0 - total num without cancer; n1 - total num with cancer
df['fit10'] = df.fit_val >= 10
df.groupby(['crc']).fit10.sum()
df.groupby('crc').size()

p0, p1, xstat, p, obs, exp = xtest(5617, 577)
1 - sci.stats.chi2.cdf(xstat, 1)


# Median age: Mann-Whitney U test
age0 = df.loc[df.crc == 0, 'age_at_fit']
age1 = df.loc[df.crc == 1, 'age_at_fit']

plt.hist(age0, bins=20)
plt.show()

plt.hist(age1, bins=20)
plt.show()

out = sci.stats.mannwhitneyu(age0, age1, use_continuity=True, alternative='two-sided')
out.statistic
out.pvalue
mann_whitney_test(df, 'age_at_fit')


# Bloods: Chi-square test
"""
low haemoglobin,16957 (33.4%),345 (52.4%)
high platelets,5318 (10.5%),139 (21.1%),g21_
high white cells,6337 (12.5%),106 (16.1%),g24_
low mean cell haemoglobin,7865 (15.5%),232 (35.2%),g27_
low mean cell volume,3206 (6.3%),123 (18.7%),g30_
"""

xtest(16957, 345)  # low HGB
xtest(5318, 139)  # high plt
xtest(6337, 106)  # high white
xtest(7865, 232)  # low MC hgb
xtest(3206, 123)  # low MCV


# Ethnicity: chi-square test
demo = pd.read_csv(r'C:\Users\5lJC\Desktop\dataprep_fitml_and_fitval\data-colofit_fu-180\demo.csv')
assert demo.shape[0] == demo.patient_id.nunique()

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
demo.ethnicity.value_counts()
demo.ethnicity.isna().sum()
demo.ethnicity = demo.ethnicity.fillna('Not known')
demo.ethnicity.value_counts()
demo = demo.merge(df[['patient_id', 'crc']], how='left')
assert demo.shape[0] == demo.patient_id.nunique()
assert demo.crc.sum() == df.crc.sum()
xtest_loop(demo, 'ethnicity')

# Age groups Chi-squared test
labs = ['0-17.9', '18-39.9', '40-49.9', '50-59.9', '60-69.9', '70-79.9', '≥80']
df['age_cat'] = pd.cut(df.age_at_fit, bins=[0,18,40,50,60,70,80,10000], labels=labs, right=False)
df.age_cat = df.age_cat.astype(str)
xtest_loop(df, 'age_cat')

# FIT groups Chi-squared test
df['fit_cat'] = pd.cut(df.fit_val, bins=[0,2,10,100,10000], 
                        labels=['0-1.9', '2-9.9', '10-99.9', '≥100'], right=False)
df.fit_cat = df.fit_cat.astype(str)
xtest_loop(df, 'fit_cat')

# FIT values Mann-Whitney test
mann_whitney_test(df, 'fit_val')

# IMD Mann-Whitney test
mann_whitney_test(demo.dropna(subset=['imdd_max']), 'imdd_max')

# IMD not known - xtest
demo['imd_not_known'] = demo.imdd_max.isna()
demo.imd_not_known.sum()
s = demo.groupby(['crc']).imd_not_known.sum()
n0_pos = s.loc[0]
n1_pos = s.loc[1]
print(n0_pos, n1_pos)
if n1_pos >= 10:
    p0, p1, xstat, p, obs, exp = xtest(n0_pos, n1_pos, return_data=True)
    1 - sci.stats.chi2.cdf(xstat, 1)
    p < 0.001
else:
    print('small count')


# Symptoms: chi squared test
dfsym = pd.read_csv(r'C:\Users\5lJC\Desktop\dataprep_fitml_and_fitval\data-colofit_fu-180\sym.csv')
repl = {'abdomass':'Abdominal mass', 
        'abdopain':'Abdominal pain', 
        'anaemia':'Anaemia',
        'bloat':'Bloating', 
        'bloodsympt':'Blood in stool', 
        'bowelhabit':'Change in bowel habit', 
        'constipation':'Constipation', 
        'diarr':'Diarrhoea',
        'fh':'Family history of colorectal cancer', 
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
dfsym['symptom'] = dfsym.category.replace(repl)
dfsym = dfsym.loc[dfsym.patient_id.isin(df.patient_id)]
dfsym = dfsym.drop_duplicates()
dfsym.shape
dfsym['symptom'].isna().sum()

tmp = df[['patient_id']].copy()
tmp = tmp.loc[~tmp.patient_id.isin(dfsym.patient_id)]
tmp['symptom'] = 'Not known'
dfsym = pd.concat(objs=[dfsym, tmp], axis=0)

sym = dfsym.symptom.drop_duplicates().sort_values()

for val in sym:
    mask = df.patient_id.isin(dfsym.loc[dfsym.symptom == val].patient_id)
    df['ind_v'] = mask
    s = df.groupby('crc').ind_v.sum()
    n0_pos = s.loc[0]
    n1_pos = s.loc[1]
    print('\n----', val, n0_pos, n1_pos)
    if n1_pos >= 10:
        p0, p1, xstat, p, obs, exp = xtest(n0_pos, n1_pos, return_data=True)
        print(xstat)
        if p < 0.001:
            print('p < 0.001')
        else:
            print(np.round(p, 3))


# Treatments: chi squared test
events = pd.read_csv(r'C:\Users\5lJC\Desktop\dataprep_fitml_and_fitval\data-colofit_fu-180\events.csv')
tx = events.loc[events.treatment == 1]
tx = tx.loc[~tx.event.isin(['colonic stent'])]
tx = tx.loc[tx.patient_id.isin(df.patient_id)]
tx['event'].isna().sum()
tx = tx[['patient_id', 'event']].drop_duplicates()

tmp = df[['patient_id']].copy()
tmp = tmp.loc[~tmp.patient_id.isin(tx.patient_id)]
tmp['event'] = 'No treatments recorded'
tx = pd.concat(objs=[tx, tmp], axis=0)

treatments = tx.event.drop_duplicates().sort_values()
for val in treatments:
    mask = df.patient_id.isin(tx.loc[tx.event == val].patient_id)
    df['ind_v'] = mask
    s = df.groupby('crc').ind_v.sum()
    n0_pos = s.loc[0]
    n1_pos = s.loc[1]
    print('\n----', val, n0_pos, n1_pos)
    if n1_pos >= 10:
        p0, p1, xstat, p, obs, exp = xtest(n0_pos, n1_pos, return_data=True)
        print(xstat)
        if p < 0.001:
            print('p < 0.001')
        else:
            print(np.round(p, 3))

