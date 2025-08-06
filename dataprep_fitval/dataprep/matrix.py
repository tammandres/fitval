"""Generate data matrix"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
from matplotlib.ticker import FixedLocator, FixedFormatter
from dataprep.bmi import max_bmi
from dataprep.bloods import bloods_high_low
from dataprep.coredata import load_coredata, load_additional_data
from dataprep.files import OutputFiles, DQFiles
from dataprep.summarise import ethnic_dict
from pathlib import Path

# Output files
f_out = OutputFiles()
dq_out = DQFiles()

DATA_MATRIX = f_out.data_matrix
BMI_DQ = dq_out.bmi_dq
TABLE_MIS = dq_out.table_mis
PLOT_MIS = dq_out.plot_mis
PLOT_MIS_IND = dq_out.plot_mis_ind

X_FILE = f_out.x
Y_FILE = f_out.y
#DEMO_INCL = 'demo_incl.csv'
#DIAGMIN_INCL = 'diagmin_incl.csv'
#FIT_INCL = 'fit_incl.csv'


def to_dummies(df, indexcol, dummycol, prefix='test', drop_first=False):
    """Generate dummy variables"""
    ind = df[[indexcol, dummycol]].set_index(indexcol)
    ind = pd.get_dummies(ind, prefix=prefix, prefix_sep='_', 
                         dummy_na=False, columns=None, sparse=False, drop_first=drop_first, dtype=int)
    ind = ind.reset_index().groupby(indexcol).max().reset_index()
    return ind


def impute_zeroind(X):
    """Impute missing values with zero, add indicator
    Used for diagnosis, procedure, prescription codes
    """
    
    # Operate on a copy just in case
    X = X.copy() 
    
    # Find columns with nan
    cols = X.columns[X.isna().sum(axis=0) > 0]
    
    # Create missing value indicators
    X_ind = X[cols].isna().astype(int)
    X_ind.columns = 'missing_' + X_ind.columns
    
    # Fill missing values with zero and add indicators
    X = X.fillna(0)
    X = pd.concat(objs=[X, X_ind], axis=1)
    
    return X


def matrix(run_path: Path, save_data: bool = True, blood_method: str = 'nearest', 
           ind_blood_hilo: bool = False, codes: bool = False, incl_sym: bool = True, mis: bool = True, incl_bmi: bool = True,
           rm_corr: bool = True, incl_ind_blood: bool = True, incl_eth: bool = True, incl_imd: bool = True, incl_bloods: bool = True,
           high_corr: float = 0.9):
    """Generate a data matrix that can be passed to prediction models"""
    pd.set_option('display.float_format', lambda x: '%.3f' % x)  # from Dan Allan @stackoverflow

    print('\n==== Converting data to tabular format for modelling ====\n')
    
    if blood_method not in ['nearest', 'max']:
        raise ValueError("blood_method must be 'nearest' or 'max'")

    # Load data
    cdata = load_coredata(run_path)
    adata = load_additional_data(run_path)

    # Indicators for whether a blood test was done
    bloods = adata.bloods
    ind_blood = to_dummies(bloods, indexcol='patient_id', dummycol='test_code', prefix='ind_blood')
    #mu = ind_blood.set_index('patient_id').mean(axis=0)
    #col_excl = mu[mu >= 0.90].index
    #ind_blood = ind_blood.loc[:, ~ind_blood.columns.isin(col_excl)]  # (exclude indicators for tests available for >= 90% of patients)?

    # Numeric blood test values (note: 'max' doesn't necessarily make sense as max can be both good or bad. Mean?)
    if blood_method == 'max':
        X_blood = bloods.groupby(['patient_id', 'test_code'])['test_result'].max().reset_index()
        X_blood['test_code'] = 'blood_' + X_blood['test_code']
        X_blood = pd.pivot(X_blood, index='patient_id', columns='test_code', values='test_result')

    elif blood_method == 'nearest':
        bloods['days_fit_to_blood_abs'] = bloods['days_fit_to_blood'].abs()
        b = bloods.sort_values(by=['patient_id', 'test_code', 'days_fit_to_blood_abs'], 
                               ascending=[True, True, True])
        X_blood = b.groupby(['patient_id', 'test_code'])[['test_result', 'days_fit_to_blood']].first().reset_index()

        days_blood_to_fit = (X_blood.days_fit_to_blood * -1).astype(int)
        print("\nDays from blood to FIT")
        print(days_blood_to_fit.describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]))

        X_blood['test_code'] = 'blood_' + X_blood['test_code']
        X_blood = pd.pivot(X_blood, index='patient_id', columns='test_code', values='test_result')

    # Low/high indicators for selected bloods (NOT USED BY DEFAULT AS REDUDANT WITH NUMERIC)
    if ind_blood_hilo:

        # Get high-low blood test values
        bloods_hilo = bloods_high_low(adata.bloods, cdata.demo, run_path=run_path, save_data=True)

        #bloods_hilo = adata.bloods_hilo
        tmp = bloods_hilo.loc[~bloods_hilo.test_result_bin.str.contains('normal')]
        print(tmp.test_result_bin.unique())

        ind_blood2 = to_dummies(tmp, indexcol='patient_id', dummycol='test_result_bin', prefix='ind')
        ind_blood2.columns = ind_blood2.columns.str.replace(' ', '_')

    # Diagnosis, procedure, prescription codes
    if codes:
        diag_codes = adata.diag_codes
        ind_diag = to_dummies(diag_codes, indexcol='patient_id', dummycol='event', prefix='ind_diagnosis')

        proc_codes = adata.proc_codes
        ind_proc = to_dummies(proc_codes, indexcol='patient_id', dummycol='event', prefix='ind_procedure')

        pres_codes = adata.pres_codes
        ind_pres = to_dummies(pres_codes, indexcol='patient_id', dummycol='event', prefix='ind_prescription')

    # Symptoms
    if incl_sym:
        #sym = cdata.sym
        #ind_sym = to_dummies(sym, indexcol='patient_id', dummycol='category', prefix='ind_symptom')
        sym_cols = ['patient_id'] + [c for c in cdata.fit.columns if c.startswith('symptom')]
        ind_sym = cdata.fit[sym_cols]

    # Demographics: sex, ethnicity, age, IMDD
    demo = cdata.demo.copy()
    demo.ethnicity = demo.ethnicity.replace(ethnic_dict)  # Simplify ethicity
    demo.ethnicity = demo.ethnicity.fillna('Not known')
    #demo.ethnicity.loc[demo.ethnicity.isin(['Not stated'])] = 'Not known'  
    demo.loc[~demo.ethnicity.isin(['White', 'Not known', "Not stated"]), 'ethnicity'] = 'Not white'  # Simplify further as very few instances of further non-white
  
    ind_sex = to_dummies(demo, indexcol='patient_id', dummycol='gender', prefix='ind_gender', drop_first=True)
    if 'ind_gender_I' in ind_sex.columns:
        ind_sex = ind_sex.drop(labels=['ind_gender_I'], axis=1)  # due to very low count
    ind_eth = to_dummies(demo, indexcol='patient_id', dummycol='ethnicity', prefix='ind_ethnicity', drop_first=True)
    X_age = demo[['patient_id', 'age_at_fit']].drop_duplicates()
    X_imdd = demo[['patient_id', 'imdd_max']].drop_duplicates()

    # FIT
    fit = cdata.fit
    X_fit = fit[['patient_id', 'fit_val']].drop_duplicates()

    # BMI
    if adata.bmi.shape[0] > 0:
        bmi, bmi_sum = max_bmi(adata.bmi)
        bmi_sum.to_csv(run_path / BMI_DQ, index=False)
        X_bmi = bmi[['patient_id', 'bmi_max']].drop_duplicates()


    # ==== Combine data sources ====

    # Create an empty matrix with patient_id indicator in first column
    subj = fit[['patient_id']].copy()
    X = fit[['patient_id']].copy() # Init

    # For these variables, do nothing with missing values
    # i.e. fit, age, deprivation, bmi, sex, ethnicity, numeric values for bloods
    dfs = [X_fit, X_age, ind_sex] 

    if incl_bloods:
        dfs += [X_blood]
    if incl_bmi:
        dfs += [X_bmi]
    if incl_eth:
        dfs += [ind_eth]
    if incl_imd:
        dfs += [X_imdd]

    for d in dfs:
        X = X.merge(d, how='left', on='patient_id')
    
    # For these variables, fill in missing value with 0
    # i.e. diagnosis, procedure, prescription codes; indicators for bloods and symptoms
    dfs = []
    
    if incl_ind_blood:
        dfs += [ind_blood]
    
    if incl_sym:
        dfs += [ind_sym] 
    
    if codes:
        dfs += [ind_diag, ind_proc, ind_pres]

    if ind_blood_hilo:
        dfs += [ind_blood2]

    for d in dfs:
        tmp = subj.merge(d, how='left', on='patient_id')
        tmp = tmp.fillna(0)
        X = X.merge(tmp, how='left', on='patient_id')
        del tmp
    
    # Add outcome (CRC)
    diagmin = cdata.diagmin
    X['crc'] = 0
    X.loc[X.patient_id.isin(diagmin.patient_id), 'crc'] = 1
    X = X[['crc'] + X.columns.drop('crc').to_list()]

    # Remove nonword characters from column names 
    cols = X.columns.to_list()
    cols = [re.sub(r'\+', r'_plus_', c) for c in cols]
    cols = [re.sub(r'\-', r'_minus_', c) for c in cols]
    cols = [re.sub(r'[^a-zA-Z0-9_]', r'_', c) for c in cols]
    X.columns = cols

    # Check
    test = X.shape[0] == X.patient_id.nunique()
    if not test:
        raise ValueError('Data matrix does not have one row per individual; check code')

    # Remove highly correlated blood tests
    #  Based on Cherry Wu @ 
    #  https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    #  Sort columns based on missing values, so when two cols are correlated, the one with more missing values is dropped
    if rm_corr:
        print('Removing highly correlated blood tests...')

        ## Bloods
        cols = X.columns[X.columns.str.startswith('blood')]
        b = X[cols]

        ## Sort columns so that ones with more missing values come last
        nmis = b.isna().sum().sort_values()
        b_sort = b.loc[:, nmis.index]

        ## Remove highly correlated columns
        ## but keep 4 core blood tests -- need to be moved forward in upper triag matrix as well
        corebloods = ['blood_HGB', 'blood_PLT', 'blood_WBC', 'blood_MCV', 'blood_CFER', 'blood_ZCRP']
        corebloods_exist = [cb for cb in corebloods if cb in b_sort.columns]
        cols_reorder = corebloods_exist + b_sort.columns[~b_sort.columns.isin(corebloods_exist)].tolist()
        b_sort = b_sort[cols_reorder]
        c = b_sort.corr(method='spearman')
        u = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
        cols_in = [col for col in u.columns if col not in corebloods]
        u = u[cols_in]
        ex = [col for col in u.columns if any(u[col].abs().dropna() >= high_corr)]
        print('Removing blood tests due to high correlation with other blood tests: {}'.format(ex))
        cols_retain = [col for col in X.columns if col not in ex]
        X = X[cols_retain]

        ## Remove certain less correlated bloods due to interfering with multiple imputation
        ## Removing total cholesterol (ZCHO) and HDL cholesterol (ZHDL), but retaining their ratio (ZHDLR)
        ex = ['blood_ZCHO', 'blood_ZHDL']
        print('Removing additional blood tests to improve multiple imputation: {}'.format(ex))
        cols_retain = [col for col in X.columns if col not in ex]
        X = X[cols_retain]
    
    # Drop indicator variables that are the same for all patients
    X_ind = X.loc[:, X.columns.str.startswith('ind')]
    tmp = X_ind.transpose().drop_duplicates().transpose()
    ex = [col for col in X_ind.columns if col not in tmp.columns]
    print('These indicator variables are identical to some other indicators, removing: {}'.format(ex))
    cols_retain = [col for col in X.columns if col not in ex]
    X = X[cols_retain]
    
    print('\nShape of data matrix: {}'.format(X.shape))
    print('\nColumns in data matrix')
    for c in X.columns:
        print(c)

    # Separate data matrix into X (predictors) and y (outcome); drop patient_id identifier
    df = X
    y = df[['crc']].copy()
    X = df.drop(labels=['patient_id', 'crc'], axis=1)

    # Explore bloods a little
    if incl_bloods:
        b = X.loc[:, X.columns.str.startswith('blood')]
        s = b.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        s = s.transpose()
        s['p99_max_ratio'] = s['99%'] / s['max'] * 100
        #s = s.sort_values(by=['p99_max_ratio'])
        s = s.round(3)
        test = s.p99_max_ratio < 5
        print('\nBlood summary')
        print(s)

    # Save
    if save_data:
        df.to_csv(run_path / DATA_MATRIX, index=False)
        X.to_csv(run_path / X_FILE, index=False)
        y.to_csv(run_path / Y_FILE, index=False)

    # Explore missingness
    if mis:
        mis, M = explore_mis(run_path, df, save_data=save_data)

    return df


def get_mis(X, suffix=''):
    n_obs = (~X.isna()).sum(axis=0)
    n = X.isna().sum(axis=0)
    p = n/X.shape[0]
    p = p.round(3)
    mis = pd.concat(objs=[n_obs, n, p], axis=1)
    mis.columns=['n_obs', 'n_mis', 'p_mis']
    mis.columns += suffix
    #mis = mis.loc[mis.n_mis>0].sort_values(by=['n_mis'], ascending=True)
    #display(mis)
    return mis


def explore_mis(run_path, df, save_data=True):
    """Explore missing values"""
    print('\n==== EXPLORING MISSING VALUES ====\n')

    # ==== Missingness by CRC status for each column ====

    # Missingness by CRC status
    mis_crc = get_mis(df.loc[df.crc==1], suffix='_crc')
    mis_nocrc = get_mis(df.loc[df.crc==0], suffix='_nocrc')
    mis = pd.concat(objs=[mis_crc, mis_nocrc], axis=1)
    mis = mis.sort_values(by=['n_mis_crc', 'n_mis_nocrc'], ascending=[True, True])

    # Exclude some indicators that don't have missing values by design 
    mask = ~mis.index.str.contains('(?:^ind|patient_id|age|crc|^fit_val)', regex=True)
    mis = mis.loc[mask]
    if save_data:
        mis_index = mis.copy().reset_index()
        mis_index.to_csv(run_path / TABLE_MIS, index=False)

    # For each column, plot percent missing in CRC and no CRC groups
    # Based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    mis = mis.sort_values(by=['n_mis_crc', 'n_mis_nocrc'], ascending=[False, False])
    labels = mis.index.to_list()
    y = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots(1,1, figsize=[8,100])#, constrained_layout=True)
    bars0 = ax.barh(y - width/2, mis.p_mis_nocrc, width, label='No colorectal cancer')
    bars1 = ax.barh(y + width/2, mis.p_mis_crc, width, label='Colorectal cancer')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set(ylim=[-1, len(labels)], xlabel='Proportion missing', ylabel='Column')
    ax.legend(loc='upper right')
    if save_data:
        plt.savefig(run_path / PLOT_MIS, dpi=150, bbox_inches='tight')
    plt.close()


    # ==== Missingness at patient_id level ====

    # Missingness at subj level
    dfsub = df.set_index(['patient_id', 'crc']).copy()
    mask = dfsub.columns.str.contains('^ind_blood|^ind_high|^ind_low', regex=True)
    dfsub = dfsub.loc[:,~mask]
    mask = dfsub.columns.str.contains('^ind_', regex=True)
    dfsub.loc[:,mask] = dfsub.loc[:,mask].replace({0:np.nan})
    #mask = ~dfsub.columns.str.contains('(?:^ind_|age|^fit_val)', regex=True)
    #dfsub = dfsub.loc[:,mask]
    M = ~dfsub.isna()
    M = M.astype(int)

    order = M.sum(axis=1).sort_values(ascending=False).index
    M = M.loc[order]

    M = M.reset_index().set_index('patient_id')

    # Group colums
    g = []
    for c in dfsub.columns:
        if c.startswith('fit_val'):
            g.append('FIT')
        elif c.startswith('age') or c.startswith('imdd') or c.startswith('bmi') or c.startswith('ind_gender')\
        or c.startswith('ind_ethnicity'):
            g.append('demographics')
        elif c.startswith('blood'):
            g.append('bloods')
        elif c.startswith('ind_diag'):
            g.append('diagnoses')
        elif c.startswith('ind_proc'):
            g.append('procedures')
        elif c.startswith('ind_pres'):
            g.append('prescriptions')
        elif c.startswith('ind_sym'):
            g.append('symptoms')
        else:
            g.append('uncategorized')
        
    tmp = np.unique(g, return_index=True)
    ticks = tmp[1]
    labs = tmp[0]
    idx = np.argsort(ticks)
    ticks, labs = ticks[idx], labs[idx]

    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].imshow(M.loc[M.crc==0].values, cmap='Greys', aspect='auto', interpolation='none')
    ax[0].xaxis.set_major_locator(FixedLocator(ticks))
    ax[0].xaxis.set_major_formatter(FixedFormatter(labs))
    ax[0].xaxis.set_tick_params(rotation=75)#, labelsize=2)
    ax[0].set(ylabel='Individual', xlabel='Data category', title='No colorectal cancer')
    ax[1].imshow(M.loc[M.crc==1].values, cmap='Greys', aspect='auto', interpolation='none')
    ax[1].xaxis.set_major_locator(FixedLocator(ticks))
    ax[1].xaxis.set_major_formatter(FixedFormatter(labs))
    ax[1].xaxis.set_tick_params(rotation=75)
    ax[1].set(ylabel='Individual', xlabel='Data category', title='Colorectal cancer')
    if save_data:
        plt.savefig(run_path / PLOT_MIS_IND, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()

    return mis, M
