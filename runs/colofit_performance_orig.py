"""Visualise colofit performance on original data"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# From Table 4 in colofit paper
data = {
 'set': ['train', 'train', 'test', 'test'],
 'model': ['colofit', 'fit', 'colofit', 'fit'],
 'pp': [21383, 22427, 18681, 23406], 
 'pp_low': [20928, 21989, 18064, 22767], 
 'pp_high': [21837, 22865, 19319, 24052],
 'tp': [1388, 1362, 1142, 1151],
 'tp_low': [1265, 1241, 983, 992],
 'tp_high': [1510, 1484, 1305, 1314],
 'ncrc': [1515, 1515, 1237, 1237]
}

df = pd.DataFrame.from_dict(data)
df['sens'] = df.tp / df.ncrc * 100
df['ppv'] = df.tp / df.pp * 100
df['pp_per_100'] = df.pp / 100000 * 100

df_fit = df.loc[df.model == 'fit'].set_index('set')
df_colofit = df.loc[df.model == 'colofit'].set_index('set')
assert all(df_fit.index == df_colofit.index)

delta_pp = (df_colofit.pp - df_fit.pp) 
red = (df_colofit.pp - df_fit.pp) / df_fit.pp * 100
dsens = df_colofit.sens - df_fit.sens

(df_colofit.pp.sum() - df_fit.pp.sum()) / df_fit.pp.sum() * 100

m = pd.DataFrame({'percent_reduction': red, 
                  'dsens': dsens,
                  'fit_pos': df_fit.pp_per_100,
                  'ppv_fit': df_fit.ppv})
m = m.reset_index()


fig, ax = plt.subplots(1, 3, figsize=(18, 6))

x = [1, 2]
ax[0].plot(x, m.percent_reduction, color='C0')
ax[0].scatter(x, m.percent_reduction, color='C0')
ax[0].axhline(0, -0.5, 2.5, color='red', linestyle='dashed')
ax[0].set(ylim=[-25, 1], xlim=[0.5, 2.5], ylabel='Percent reduction in referrals relative to FIT',
          title='Reduction in referrals')
ax[0].set_xticks(x)
ax[0].set_xticklabels(['Development data\n(nov 2016 - nov 2021)', 'Validation data\n(dec 2021 - nov 2022)'])

ax[1].plot(x, m.dsens, color='C0')
ax[1].scatter(x, m.dsens, color='C0')
ax[1].axhline(0, -0.5, 2.5, color='red', linestyle='dashed')
ax[1].set(ylim=[-3, 3], xlim=[0.5, 2.5], ylabel='Percent cancers gained or missed relative to FIT',
          title='Percent cancers gained/missed')
ax[1].set_xticks(x)
ax[1].set_xticklabels(['Development data\n(nov 2016 - nov 2021)', 'Validation data\n(dec 2021 - nov 2022)'])

ax[2].plot(x, m.ppv_fit, color='C0')
ax[2].scatter(x, m.ppv_fit, color='C0')
ax[2].set(ylim=[0, 10], xlim=[0.5, 2.5], ylabel='Positive predictive value (PPV)',
          title='PPV of FIT ≥ 10 µg/g (%)')
ax[2].set_xticks(x)
ax[2].set_xticklabels(['Development data\n(nov 2016 - nov 2021)', 'Validation data\n(dec 2021 - nov 2022)'])

for i in [0, 1, 2]:
    ax[i].grid(which='major', alpha=0.5)

out_path = Path(r'/Users/andres/Desktop/DPhil/4. Projects/project_colofit/fitval_code_export_nov2024/fitval/')
plt.savefig(out_path / 'colofit_performance_full.png', dpi=300)
plt.close()    



pointsize=48
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax = [ax]

x = [1, 2]
ax[0].plot(x, m.percent_reduction, color='C0')
ax[0].scatter(x, m.percent_reduction, color='C0', s=pointsize)
ax[0].axhline(0, -0.5, 2.5, color='red', linestyle='dashed')
ax[0].set(ylim=[-25, 1], xlim=[0.5, 2.5], ylabel='Percent reduction in referrals relative to FIT')
ax[0].set_xticks(x)
ax[0].set_xticklabels(['Model development data\n(Nov 2016 - Nov 2021)', 'Internal validation data\n(Dec 2021 - Nov 2022)'])
ax[0].grid(which='major', alpha=0.5)

out_path = Path(r'/Users/andres/Desktop/DPhil/4. Projects/project_colofit/fitval_code_export_nov2024/fitval/')
plt.savefig(out_path / 'colofit_performance_simple.png', dpi=300)
plt.close()    



pointsize=48
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax = [ax]

x = [1, 2]
ax[0].plot(x, m.percent_reduction, color='C0', label='Nottingham data')
ax[0].scatter(x, m.percent_reduction, color='C0', s=pointsize)

ax[0].plot(x, [-4.16, -19.15], color='C1', label='Oxford data')
ax[0].scatter(x, [-4.16, -19.15], color='C1', s=pointsize)

ax[0].axhline(0, -0.5, 2.5, color='red', linestyle='dashed')
ax[0].set(ylim=[-25, 1], xlim=[0.5, 2.5])
ax[0].set_ylabel('Percent reduction in referrals relative to FIT')
ax[0].set_xticks(x)
ax[0].set_xticklabels(['Model derivation data period\n(Nov 2016 - Nov 2021)', 'Model internal validation data period\n(Dec 2021 - Nov 2022)'])
ax[0].grid(which='major', alpha=0.5)
ax[0].legend(frameon=False, loc=(0.68, 0.8))

out_path = Path(r'/Users/andres/Desktop/DPhil/4. Projects/project_colofit/fitval_code_export_nov2024/fitval/')
plt.savefig(out_path / 'colofit_performance_nott-and-ox.png', dpi=300)
plt.close()    



pointsize=48
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))

x = [1, 2]
ax[0].plot(x, m.percent_reduction, color='C0', label='Nottingham data')
ax[0].scatter(x, m.percent_reduction, color='C0', s=pointsize)
ax[0].axhline(0, -0.5, 2.5, color='red', linestyle='dashed')
ax[0].set(ylim=[-25, 1], xlim=[0.5, 2.5], ylabel='Percent reduction in referrals relative to FIT')
ax[0].set_xticks(x)
ax[0].set_xticklabels(['Model development data\n(Nov 2016 - Nov 2021)', 'Internal validation data\n(Dec 2021 - Nov 2022)'])
ax[0].grid(which='major', alpha=0.5)
ax[0].set_title('A. Performance on Nottingham data')
#ax[0].legend(frameon=False, loc=(0.6, 0.8))

ax[1].plot(x, m.percent_reduction, color='C0', label='Nottingham data')
ax[1].scatter(x, m.percent_reduction, color='C0', s=pointsize)

ax[1].plot(x, [-4.16, -19.15], color='C1', label='Oxford data')
ax[1].scatter(x, [-4.16, -19.15], color='C1', s=pointsize)

ax[1].axhline(0, -0.5, 2.5, color='red', linestyle='dashed')
ax[1].set(ylim=[-25, 1], xlim=[0.5, 2.5])
ax[1].set_ylabel('Percent reduction in referrals relative to FIT')
ax[1].set_xticks(x)
ax[1].set_xticklabels(['Nov 2016 - Nov 2021 (Nottingham)\nJan 2017 - Nov 2021 (Oxford)', 'Dec 2021 - Nov 2022'])
ax[1].grid(which='major', alpha=0.5)
ax[1].legend(frameon=False, loc=(0.6, 0.8))
ax[1].set_title('B. Performance on Nottingham and Oxford data\nwithin a similar time period')

out_path = Path(r'/Users/andres/Desktop/DPhil/4. Projects/project_colofit/fitval_code_export_nov2024/fitval/')
plt.savefig(out_path / 'colofit_performance_simple_and-nott-ox.png', dpi=300)
plt.savefig(out_path / 'colofit_performance_simple_and-nott-ox.svg', dpi=300)
plt.close()    



# Add