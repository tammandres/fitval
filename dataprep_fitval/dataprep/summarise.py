import numpy as np
import pandas as pd
from dataprep.bmi import max_bmi
from dataprep.bloods import bloods_high_low
from dataprep.coredata import load_coredata, load_additional_data, CoreData, AdditionalData
from dataprep.files import OutputFiles
from pathlib import Path

f_out = OutputFiles()
SUM_TABLE = f_out.sum_table


# ==== Functions for summarising data ====
def summarise_cat(df, col, nsub, nanfill = 'Not known', order = None, hide_low_count = True,
                  hide_only_count_itself = False):
    e = df.fillna(nanfill).groupby(col)['patient_id'].nunique()
    p = df.fillna(nanfill).groupby(col)['patient_id'].nunique()/nsub*100
    p = p.round(1)

    # Identify all counts less than 10, e.g. in "1, 2, 5, 11, 13" the mask covers "1, 2, 5"
    e = e.sort_values()
    mask = e < 10  # 
    
    # Identify where cumulative sum is less than 10, and include the next highest value
    # e.g. in "1, 2, 5, 11, 13", cumsum is "1, 3, 8, 19, 32", and mask is "True, True, True, True, False"
    cumsum = e.cumsum()
    cumsum_mask = (cumsum < 10) 
    if cumsum_mask.any():
        cumsum_mask.loc[(~cumsum_mask).idxmax()] = True 
    
    # Mask to hide all values less than 10, and if their cumsum is less than 10 we also hide one extra value
    # to make it not possible to infer that a certain set of values was available for less than 10 patients
    if hide_only_count_itself:
        mask_hide = mask
    else:
        mask_hide = mask | cumsum_mask
    
    e = e.astype(str) + ' (' + p.astype(str) + '%)'
    if mask_hide.any() and hide_low_count: 
        e[mask_hide] = 'Not Available'  # All categories where counts sum to less than 10 in total are hidden

    if order is not None:
        e = e.loc[order]
    
    #e = e.reset_index()
    #e['group'] = col
    #e = e.set_index(['group', col])
    return e


def summarise_con(df, col, nsub, nanfill = 'Not known'):
    e = pd.DataFrame()

    # Get data
    t = df[col].quantile([0, 0.25, 0.5, 0.75, 1])
    vmin = np.round(t[0], 1)
    med = np.round(t[0.5],1)
    p25 = np.round(t[0.25],1)
    p75 = np.round(t[0.75],1)
    vmax = np.round(t[1], 1)
    #iqr = np.round(t[0.75] - t[0.25],1)
    #a = str(med) + ' (' + str(iqr) + ')'

    # Median, 25th and 75th percentile
    a = str(med) + ' (' + str(p25) + ', ' + str(p75) + ')'
    a = pd.Series(a, index=['Median (25th, 75th percentile)'])
    e = pd.concat(objs=[e,a], axis=0)

    # Minimum and maximum
    a = str(vmin) + ', ' + str(vmax)
    a = pd.Series(a, index=['Min, max'])
    e = pd.concat(objs=[e,a], axis=0)
    
    # Missingness count
    n = df[col].isna().sum()
    if nanfill is not None and n > 0:
        if n < 10:
            a = pd.Series('<10', index=[nanfill])
            e = pd.concat(objs=[e,a], axis=0)
        else:
            p = np.round(n/nsub*100,1)
            a = str(n) + ' (' + str(p) + '%)'
            a = pd.Series(a, index=[nanfill])
            e = pd.concat(objs=[e,a], axis=0)
    
    #e = e.reset_index()
    #e['group'] = col
    #e = e.set_index(['group', 'index'])
    
    return e


# For simplifying ethnicity
# https://datadictionary.nhs.uk/data_elements/ethnic_category.html
ethnic_dict_letter = {
    'A':'White',
    'B':'White',
    'C':'White',
    'D':'Mixed',
    'E':'Mixed',
    'F':'Mixed',
    'G':'Mixed',
    'H':'Asian', # or Asian British',
    'J':'Asian', # or Asian British',
    'K':'Asian', # or Asian British',
    'L':'Asian', # or Asian British',
    'M':'Black', # or Black British',
    'N':'Black', # or Black British',
    'P':'Black', # or Black British',
    'R':'Other Ethnic Groups',
    'S':'Other Ethnic Groups',
    'Z':'Not stated',
    '99':'Not known'}

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


def _add_group(t, s):
    i = t.group.str.extract('(\d+)', expand=False).dropna().astype(int).max() + 1
    g = 'g' + '{:02d}_'.format(i)
    s['group'] = g
    return s


def summary_table(run_path: Path, save_data: bool = True, blood_method: str = 'nearest',
                  cdata: CoreData = None, adata: AdditionalData = None):
    """Summarise data"""
    print('\nSummarising data...')

    # Load data from disk if not inputted
    if cdata is None:
        cdata = load_coredata(run_path)
    if adata is None:
        adata = load_additional_data(run_path)

    # Get tables to be summarised
    fit = cdata.fit  # FIT values
    diag = cdata.diagmin  # Presence of CRC and its date
    demo = cdata.demo  # Demographics
    dfsym = cdata.sym  # Clinical symptoms, e.g. abdominal pain
    events = cdata.events  # Event logs
    tx = events.loc[events.treatment == 1]  # Treatment events and dates
    dcode = adata.diag_codes  # Diagnosis codes
    dproc = adata.proc_codes  # Procedure codes
    dpres = adata.pres_codes  # Prescription codes

    # Blood test results to be summarised: get one value per patient first
    blood = adata.bloods  

    if blood_method == 'max':
        blood = blood.groupby(['patient_id', 'test_code'])['test_result'].max().reset_index()

    elif blood_method == 'nearest':
        blood['days_fit_to_blood_abs'] = blood['days_fit_to_blood'].abs()
        blood = blood.sort_values(by=['patient_id', 'test_code', 'days_fit_to_blood_abs'], ascending=[True, True, True])
        blood = blood.groupby(['patient_id', 'test_code'])['test_result'].first().reset_index()

    bsub = bloods_high_low(adata.bloods, cdata.demo, run_path=None, save_data=False)

    # Body mass index
    bmi, __ = max_bmi(adata.bmi)  

    # T-stage from TNM system
    stage = diag.rename(columns={'T': 'T_stage'}).copy()
    stage.T_stage = stage.T_stage.str[0]  

    # Copy of demographics table with I -> nan
    d = demo.copy()  
    d.gender = d.gender.replace({'I': np.nan})


    #-----------------------------------
    # Initialise empty table    
    t = pd.DataFrame()  

    # Get patient groups (no CRC and CRC) and counts within groups
    subj0 = fit.loc[~fit.patient_id.isin(diag.patient_id)].patient_id.unique()
    n0 = len(subj0)
    subj1 = diag.patient_id.unique()
    n1 = len(subj1)
    cols = ['No colorectal cancer', 'Colorectal cancer']

    s = pd.DataFrame([[n0, n1]], columns=cols, index=['Number of patients'])
    s['group'] = 'g00_'
    t = pd.concat(objs=[t, s], axis=0)

    # Age
    s = pd.DataFrame([['','']], columns=cols, index=['Age'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    labs = ['0-17.9', '18-39.9', '40-49.9', '50-59.9', '60-69.9', '70-79.9', '≥80']
    d['age_cat'] = pd.cut(d.age_at_fit, bins=[0,18,40,50,60,70,80,10000], labels=labs, right=False)
    d.age_cat = d.age_cat.astype(str)

    s0 = summarise_cat(d.loc[d.patient_id.isin(subj0)], 'age_cat', n0)
    s1 = summarise_cat(d.loc[d.patient_id.isin(subj1)], 'age_cat', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    s0 = summarise_con(d.loc[d.patient_id.isin(subj0)], 'age_at_fit', n0)
    s1 = summarise_con(d.loc[d.patient_id.isin(subj1)], 'age_at_fit', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.index = ['Age, median (25th, 75th percentile)', 'Age, min and max']
    s.columns = cols
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    # Sex
    s = pd.DataFrame([['','']], columns=cols, index=['Gender'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    s0 = summarise_cat(d.loc[d.patient_id.isin(subj0)], 'gender', n0)
    s1 = summarise_cat(d.loc[d.patient_id.isin(subj1)], 'gender', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    # Ethnicity
    d.ethnicity = d.ethnicity.replace(ethnic_dict)
    
    s = pd.DataFrame([['','']], columns=cols, index=['Ethnicity'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    s0 = summarise_cat(d.loc[d.patient_id.isin(subj0)], 'ethnicity', n0)
    s1 = summarise_cat(d.loc[d.patient_id.isin(subj1)], 'ethnicity', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)

    reorder = ['Asian', 'Black', 'Mixed', 'Other Ethnic Groups', 'White', 'Not stated', 'Not known']
    s = s.loc[reorder]
    
    t = pd.concat(objs=[t, s], axis=0)

    # Index of multiple deprivation
    s = pd.DataFrame([['','']], columns=cols, index=['IMDD'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    s0 = summarise_con(d.loc[d.patient_id.isin(subj0)], 'imdd_max', n0)
    s1 = summarise_con(d.loc[d.patient_id.isin(subj1)], 'imdd_max', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    # FIT
    s = pd.DataFrame([['','']], columns=cols, index=[ 'FIT (μg Hb/g)'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    fit['fit_cat'] = pd.cut(fit.fit_val, bins=[0,2,10,100,10000], 
                            labels=['0-1.9', '2-9.9', '10-99.9', '≥100'], right=False)
    fit.fit_cat = fit.fit_cat.astype(str)
    s0 = summarise_cat(fit.loc[fit.patient_id.isin(subj0)], 'fit_cat', n0)
    s1 = summarise_cat(fit.loc[fit.patient_id.isin(subj1)], 'fit_cat', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)

    reorder = ['0-1.9', '2-9.9', '10-99.9', '≥100']
    s = s.loc[reorder]

    t = pd.concat(objs=[t, s], axis=0)

    s0 = summarise_con(fit.loc[fit.patient_id.isin(subj0)], 'fit_val', n0)
    s1 = summarise_con(fit.loc[fit.patient_id.isin(subj1)], 'fit_val', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    # Symptoms
    print(dfsym.category.unique())
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
    dfsym2 = fit[['patient_id']].drop_duplicates().merge(dfsym, how='left', on='patient_id')

    s = pd.DataFrame([['','']], columns=cols, index=['Symptoms - GP reported'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    s0 = summarise_cat(dfsym2.loc[dfsym2.patient_id.isin(subj0)], 'symptom', n0)
    s1 = summarise_cat(dfsym2.loc[dfsym2.patient_id.isin(subj1)], 'symptom', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)

    if 'Not known' in s.index:
        mask = ['Not known' == v for v in s.index]
        mask_inv = [not v for v in mask]
        s = pd.concat(objs=[s.loc[mask_inv], s.loc[mask]], axis=0)
    t = pd.concat(objs=[t, s], axis=0)


    # ==== Bloods (selected) ====

    # How to deal w missing values here?
    s = pd.DataFrame([['','']], columns=cols, index=['SELECTED BLOODS'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    repl = {'HGB':'haemoglobin', 'PLT':'platelets', 'WBC':'white cells', 'MCH':'mean cell haemoglobin',
            'MCV':'mean cell volume', 'CFER':'serum ferritin', 'ZCRP':'C-reactive protein',
            'CRPP':'C-reactive protein'}

    blood_select = blood.loc[blood.test_code.isin(bsub.test_code)].copy()
    blood_select['name'] = blood_select.test_code.replace(repl)
    bsub['name'] = bsub.test_code.replace(repl)
    for c in bsub.name.unique():
        #print(c)
        
        b = blood_select.loc[blood_select.name==c].copy()
        
        s = pd.DataFrame([['','']], columns=cols, index=[c.upper()])
        s = _add_group(t, s)
        t = pd.concat(objs=[t, s], axis=0)
        
        s0 = summarise_con(b.loc[b.patient_id.isin(subj0)], 'test_result', n0)
        s1 = summarise_con(b.loc[b.patient_id.isin(subj1)], 'test_result', n1)
        s = pd.concat(objs=[s0,s1], axis=1)
        s.columns = cols
        s.index = [c + ', median (25th, 75th percentile)', c + ', min and max']
        s = _add_group(t, s)
        t = pd.concat(objs=[t, s], axis=0)
        
        b = bsub.loc[bsub.name==c]
        b = fit[['patient_id']].merge(b, how='left', on='patient_id')
        s0 = summarise_cat(b.loc[b.patient_id.isin(subj0)], 'test_result_bin', n0, nanfill='Not known')
        s1 = summarise_cat(b.loc[b.patient_id.isin(subj1)], 'test_result_bin', n1, nanfill='Not known')
        s = pd.concat(objs=[s0,s1], axis=1)
        s.columns = cols
        s = _add_group(t, s)
        
        if 'Not known' in s.index:
            reorder = s.index.drop('Not known').to_list() + ['Not known']
            s = s.loc[reorder]
        t = pd.concat(objs=[t, s], axis=0)


    # BMI
    s = pd.DataFrame([['','']], columns=cols, index=['BMI'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    d = fit[['patient_id']].merge(bmi, how='left')
    s0 = summarise_con(d.loc[d.patient_id.isin(subj0)], 'bmi_max', n0)
    s1 = summarise_con(d.loc[d.patient_id.isin(subj1)], 'bmi_max', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    # Count blood tests
    s = pd.DataFrame([['','']], columns=cols, index=['Number of codes per patient'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)
    d = blood.groupby('patient_id')[['test_code']].nunique().reset_index().rename(columns={'test_code':'n_test'})

    s0 = summarise_con(d.loc[d.patient_id.isin(subj0)], 'n_test', n0)
    s1 = summarise_con(d.loc[d.patient_id.isin(subj1)], 'n_test', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)
    #s.index = ['Number of unique blood test codes, median (25th, 75th percentile)',
    #           'Number of unique blood test codes, min and max']
    t = pd.concat(objs=[t, s], axis=0)

    # Count diagnosis codes
    if not dcode.empty:
        d = dcode.groupby('patient_id')[['event']].nunique().reset_index()
        s0 = summarise_con(d.loc[d.patient_id.isin(subj0)], 'event', n0)
        s1 = summarise_con(d.loc[d.patient_id.isin(subj1)], 'event', n1)
        s = pd.concat(objs=[s0,s1], axis=1)
        s.columns = cols
        s = _add_group(t, s)
        s.index = ['Number of unique diagnosis codes, median (25th, 75th percentile)',
                   'Number of unique diagnosis codes, min and max']
        t = pd.concat(objs=[t, s], axis=0)

    # Count procedure codes
    if not dproc.empty:
        d = dproc.groupby('patient_id')[['event']].nunique().reset_index()
        s0 = summarise_con(d.loc[d.patient_id.isin(subj0)], 'event', n0)
        s1 = summarise_con(d.loc[d.patient_id.isin(subj1)], 'event', n1)
        s = pd.concat(objs=[s0,s1], axis=1)
        s.columns = cols
        s = _add_group(t, s)
        s.index = ['Number of unique procedure codes, median (25th, 75th percentile)',
                'Number of unique procedure codes, min and max']
        t = pd.concat(objs=[t, s], axis=0)

    # Count prescription codes
    if not dpres.empty:
        d = dpres.groupby('patient_id')[['event']].nunique().reset_index()
        s0 = summarise_con(d.loc[d.patient_id.isin(subj0)], 'event', n0)
        s1 = summarise_con(d.loc[d.patient_id.isin(subj1)], 'event', n1)
        s = pd.concat(objs=[s0,s1], axis=1)
        s.columns = cols
        s = _add_group(t, s)
        s.index = ['Number of unique prescription codes, median (25th, 75th percentile)',
                'Number of unique prescription codes, min and max']
        t = pd.concat(objs=[t, s], axis=0)


    # ==== Treatments ====
    tx2 = tx.loc[~tx.event.isin(['colonic stent'])]
    tx2 = fit[['patient_id']].drop_duplicates().merge(tx2, how='left', on='patient_id')
    s = pd.DataFrame([['','']], columns=cols, index=['CRC-relevant treatments'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    s0 = summarise_cat(tx2.loc[tx2.patient_id.isin(subj0)], 'event', n0, nanfill='No treatments recorded')
    s1 = summarise_cat(tx2.loc[tx2.patient_id.isin(subj1)], 'event', n1, nanfill='No treatments recorded')
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)

    t = pd.concat(objs=[t, s], axis=0)


    # ==== T-stage ===
    s = pd.DataFrame([['','']], columns=cols, index=['T stage'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    stage2 = fit[['patient_id']].drop_duplicates().merge(stage, how='left', on='patient_id')

    s0 = summarise_cat(stage2.loc[stage2.patient_id.isin(subj0)], 'T_stage', n0)
    s0.iloc[0] = '-'
    s1 = summarise_cat(stage2.loc[stage2.patient_id.isin(subj1)], 'T_stage', n1)
    s = pd.concat(objs=[s0,s1], axis=1)
    s.columns = cols
    s = _add_group(t, s)

    if 'Not known' in s.index:
        reorder = s.index.drop('Not known').to_list() + ['Not known']
        s = s.loc[reorder]
        
    t = pd.concat(objs=[t, s], axis=0)


    # ==== Replace nan ====
    t = t.fillna('-')
    t = t.reset_index()
    t = t.rename(columns={'index': 'characteristic'})


    if save_data:
        t.to_csv(run_path / SUM_TABLE, index=False)

    return t


def summary_table_bloodsonly(run_path: Path, cdata: CoreData = None, adata: AdditionalData = None):
    """Temporary function to only summarise blood test results.
    For high/low/normal results, it hides only the low count and not another category:
    given that the percentages do not sum to 100, there is no risk of a low count 
    being inferred from the total."""
    print('\nSummarising data...')

    # Load data from disk if not inputted
    if cdata is None:
        cdata = load_coredata(run_path)
    if adata is None:
        adata = load_additional_data(run_path)

    # Get tables to be summarised
    fit = cdata.fit  # FIT values
    diag = cdata.diagmin  # Presence of CRC and its date

    # Get blood result nearest to FIT for each person
    blood = adata.bloods  
    blood['days_fit_to_blood_abs'] = blood['days_fit_to_blood'].abs()
    blood = blood.sort_values(by=['patient_id', 'test_code', 'days_fit_to_blood_abs'], ascending=[True, True, True])
    blood = blood.groupby(['patient_id', 'test_code'])['test_result'].first().reset_index()

    # Get low/high/normal blood results
    bsub = bloods_high_low(adata.bloods, cdata.demo, run_path=None, save_data=False)

    # Initialise empty table    
    t = pd.DataFrame()  

    # Get patient groups (no CRC and CRC) and counts within groups
    subj0 = fit.loc[~fit.patient_id.isin(diag.patient_id)].patient_id.unique()
    n0 = len(subj0)
    subj1 = diag.patient_id.unique()
    n1 = len(subj1)
    cols = ['No colorectal cancer', 'Colorectal cancer']

    s = pd.DataFrame([[n0, n1]], columns=cols, index=['Number of patients'])
    s['group'] = 'g00_'
    t = pd.concat(objs=[t, s], axis=0)

    # Summarise bloods (selected)
    s = pd.DataFrame([['','']], columns=cols, index=['SELECTED BLOODS'])
    s = _add_group(t, s)
    t = pd.concat(objs=[t, s], axis=0)

    repl = {'HGB':'haemoglobin', 'PLT':'platelets', 'WBC':'white cells', 'MCH':'mean cell haemoglobin',
            'MCV':'mean cell volume', 'CFER':'serum ferritin', 'ZCRP':'C-reactive protein',
            'CRPP':'C-reactive protein'}

    blood_select = blood.loc[blood.test_code.isin(bsub.test_code)].copy()
    blood_select['name'] = blood_select.test_code.replace(repl)
    bsub['name'] = bsub.test_code.replace(repl)
    for c in bsub.name.unique():
        #print(c)
        
        b = blood_select.loc[blood_select.name==c].copy()
        
        s = pd.DataFrame([['','']], columns=cols, index=[c.upper()])
        s = _add_group(t, s)
        t = pd.concat(objs=[t, s], axis=0)
        
        s0 = summarise_con(b.loc[b.patient_id.isin(subj0)], 'test_result', n0)
        s1 = summarise_con(b.loc[b.patient_id.isin(subj1)], 'test_result', n1)
        s = pd.concat(objs=[s0,s1], axis=1)
        s.columns = cols
        s.index = [c + ', median (25th, 75th percentile)', c + ', min and max']
        s = _add_group(t, s)
        t = pd.concat(objs=[t, s], axis=0)
        
        b = bsub.loc[bsub.name==c]
        b = fit[['patient_id']].merge(b, how='left', on='patient_id')
        s0 = summarise_cat(b.loc[b.patient_id.isin(subj0)], 'test_result_bin', n0, nanfill='Not known', hide_only_count_itself=True)
        s1 = summarise_cat(b.loc[b.patient_id.isin(subj1)], 'test_result_bin', n1, nanfill='Not known', hide_only_count_itself=True)
        s = pd.concat(objs=[s0,s1], axis=1)
        s.columns = cols
        s = _add_group(t, s)
        
        if 'Not known' in s.index:
            reorder = s.index.drop('Not known').to_list() + ['Not known']
            s = s.loc[reorder]
        t = pd.concat(objs=[t, s], axis=0)

    # Replace nan
    t = t.fillna('-')
    t = t.reset_index()
    t = t.rename(columns={'index': 'characteristic'})

    return t
