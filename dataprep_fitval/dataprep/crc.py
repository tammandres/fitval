"""Get colorectal cancer (CRC) indicator, its date, and treatments usually given for CRC

1) CRC date is identified using pathology reports, inpatient diagnosis codes, outpatient diagnosis codes
2) Earliest date is chosen among pathology, inpatient, outpatient dates
3) If a CRC-like treatment occurs within 180 days of the earliest date, then treatment date set as CRC date

Q: should imaging reports be included? Not atm as may not be as reliable source of information
"""
import numpy as np
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
#import textmining.utils as ut
#from textmining.reports import get_crc_reports
from constants import DATA_DIR, DATETIME_FORMAT, DATE_FORMAT, PARQUET
from dataprep.files import InputFiles, OutputFiles, DQFiles
from dataprep import dq
from pathlib import Path

f_in = InputFiles()
f_out = OutputFiles()
dq_out = DQFiles()

# Input files
HIST_FILE = f_in.hist
IMG_FILE = f_in.imaging
INPAT_DIAG_FILE = f_in.inpat_diag
OUTPAT_DIAG_FILE = f_in.outpat_diag
INPAT_PROC_FILE = f_in.inpat_proc
OUTPAT_PROC_FILE = f_in.outpat_proc
CHEMO_ADMIN_FILE = f_in.chemo_admin
CHEMO_SUM_FILE = f_in.chemo_sum
RADIO_FILE = f_in.radio
CRC_FROM_PATH_FILE = f_in.crc_from_path
CRC_MATCHES_FILE = f_in.crc_matches
OUTPAT_FILE = f_in.outpat
INFOFLEX_FILE = f_in.infoflex

# Output files
DIAGMIN_OUT = f_out.diagmin
DIAG_OUT = f_out.diag
REPORTS_CRC_PATH = dq_out.reports_crc_path
MATCHES_CRC_PATH = dq_out.matches_crc_path
PLOT_DAYS_FIT_CRC = dq_out.plot_days_fit_crc

# ==== CRC and its date ====
#region
def crc_and_tx(fit: pd.DataFrame, run_path: Path, data_path: Path = DATA_DIR, 
               save_data: bool = True, testmode: bool = False,
               days_treatment_before_diag: int = 180, min_date: bool = False,
               infoflex: bool = False):
    """Get earliest CRC date for patients with FIT values

    Args
        fit: dataframe with patient_id identifiers and earliest FIT values
        run_path: where to save results
        save_data: if True, outputs are saved to disk
        testmode: if True, code is run on a small number of reports
        days_treatment_before_diag: treatment date is set to be CRC date 
            if occurs within that many days before CRC dates identified from pathology, inpatient, outpatient data

    Outputs
        diagmin: earliest CRC date for each individual
        diag: all records of CRC (from inpatient, outpatient, pathology, and treatment data*), with dates
        reports: all pathology reports, those that contain CRC are marked
        matches: matches for CRC in pathology reports
        tx: treatments relevant for CRC, and their dates

    * If any CRC-like treatment occurred within 180 days of inpat, outpat, or path date, 
      then this was set as CRC date 
    """
    print('\n==== PREPARING CRC AND TREATMENT DATA ====\n')
    diag = pd.DataFrame()

    if not infoflex:
        # CRC dates from pathology reports
        reports, diag_path, matches = crc_pathology(fit, data_path=data_path, testmode=testmode, min_date=min_date)
        diag = pd.concat(objs=[diag, diag_path], axis=0)

        # CRC dates from inpatient and outpatient data
        diag_inpat_outpat = crc_inpat_outpat(fit, min_date=min_date)
        diag = pd.concat(objs=[diag, diag_inpat_outpat], axis=0)
        diag = diag.drop_duplicates().reset_index(drop=True)
        diag.diagnosis_source.value_counts()
    else:
        df = pd.read_parquet(data_path / INFOFLEX_FILE, columns= ['patient_id', 'date_of_diagnosis', 'diagnosis_code_icd10'])
        df = df.dropna(subset='date_of_diagnosis')

        # Retain patients with FIT
        df = df.loc[df.patient_id.isin(fit.patient_id)]

        # Rename columns
        df = df.rename(columns={'date_of_diagnosis': 'diagnosis_date'})
        
        df.diagnosis_date = df.diagnosis_date.dt.normalize()

        ## Retain patients with CRC diag codes
        mask = df.diagnosis_code_icd10.astype(str).str.lower().str.contains('^c18|^c19|^c20', regex=True)
        df = df.loc[mask]

        ## Assign diagnosis source
        df['diagnosis_source'] = 'infoflex'
        print('\nNumber of patients with C18-C20 diagnosis from infoflex: {}'.format(df.patient_id.nunique()))

        # Get minimum date?
        if min_date:
            idx = df.groupby('patient_id')['diagnosis_date'].idxmin()
            tmp = df.loc[idx]
            tmp = tmp[['patient_id', 'diagnosis_date', 'diagnosis_source', 'diagnosis_code_icd10']]
        else:
            tmp = df[['patient_id', 'diagnosis_date', 'diagnosis_source', 'diagnosis_code_icd10']]
        diag = pd.concat(objs=[diag, tmp], axis=0).reset_index(drop=True)
        reports = pd.DataFrame()
        matches = pd.DataFrame()

    # Treatment data
    tx = treatments(fit, data_path=data_path)

    # Minimum CRC date
    diagmin = earliest_crc_date(diag, tx, fit, run_path, days_treatment_before_diag=days_treatment_before_diag)

    if save_data:
        diagmin.to_csv(run_path / DIAGMIN_OUT, index=False)
        diag.to_csv(run_path / DIAG_OUT, index=False)
        reports.to_csv(run_path / REPORTS_CRC_PATH, index=False)
        matches.to_csv(run_path / MATCHES_CRC_PATH, index=False)

    return diagmin, diag, reports, matches, tx 
#endregion

# ==== CRC and its date from PATHOLOGY REPORTS ====
#region
def crc_pathology(fit, datecol: str = 'received_date', data_path: Path = DATA_DIR,  # datecol=date_received
                  testmode: bool = False, min_date: bool = True):
    """Identify pathology reports that describe CRC, and get earliest date of CRC for each individual"""
    
    use_preprocessed_data = True

    if use_preprocessed_data:
        
        # Read extracted features for each report
        if PARQUET:
            df = pd.read_parquet(data_path / CRC_FROM_PATH_FILE)
            ind_cols = ['crc_nlp', 'crc_snomed']
            df[ind_cols] = df[ind_cols].astype(int)
        else:
            df = pd.read_csv(data_path / CRC_FROM_PATH_FILE)
        print('\nNumber of reports where CRC was detected: ' + str(df.shape[0]))
        print('\nColumns: {}'.format(df.columns))
        print('\nMissingness: \n{}'.format(df.isna().sum(axis=0)))

        # Retain patients who have FIT values
        df = df.loc[df.patient_id.isin(fit.patient_id)]
        print('\nNumber of reports for patients with FIT values: ' + str(df.shape[0]))

        # Retain reports describing current CRC 
        df = df.loc[df.crc_nlp==1]

        # Date to datetime
        if not PARQUET:
            dq.check_date_format(df, datecols=[datecol], formats=['%Y-%m-%d'])
        df[datecol] = pd.to_datetime(df[datecol], format='%Y-%m-%d')

        # Read matches corresponding to each report
        # Currently, hard to say which report the matches correspond to --> need to fix this
        # but this is not crucial atm.
        if PARQUET:
            matches = pd.read_parquet(data_path / CRC_MATCHES_FILE)
        else:
            matches = pd.read_csv(data_path / CRC_MATCHES_FILE)
        
        dfpath = df
    else:
        raise NotImplementedError

        # ---- Get pathology reports ----
        print('\n---- Reading pathology reports')

        # Read data
        if PARQUET:
            df = pd.read_parquet(data_path / HIST_FILE)
        else:
            df = pd.read_csv(data_path / HIST_FILE, sep='|')
        print('\nNumber of pathology reports: ' + str(df.shape[0]))
        print('\nColumns: {}'.format(df.columns))
        print('\nMissingness: \n{}'.format(df.isna().sum(axis=0)))

        df = df.rename(columns={'redacted_imaging_report': 'safe_report', 'salted_master_patient_id': 'patient_id'})
        df = df[['patient_id', 'safe_report', datecol, 'snomed_t', 'snomed_m']]

        # Retain patients who have FIT values
        df = df.loc[df.patient_id.isin(fit.patient_id)]
        print('\nNumber of reports for patients with FIT values: ' + str(df.shape[0]))

        # Date to datetime
        if not PARQUET:
            dq.check_date_format(df, datecols=[datecol], formats=['%Y-%m-%d'])
            df[datecol] = pd.to_datetime(df[datecol], format='%Y-%m-%d')

        if testmode:
            nsamp = int(min(df.shape[0], 1000))
            df = df.sample(nsamp, random_state=42)


        # ---- Identify if reports talk about current colorectal cancer according to textmining ----
        print('\n---- Identifying reports that discuss current CRC')
        _, matches = get_crc_reports(df, col='safe_report', verbose=False, add_subj_to_matches=True, 
                                    subjcol='patient_id', negation_bugfix=True)
        df['row'] = np.arange(df.shape[0])
        df['crc_nlp'] = 0
        matches_incl = matches.loc[matches.exclusion_indicator == 0]
        df.loc[df.row.isin(matches_incl.row), 'crc_nlp'] = 1


        # ---- Identify if reports talk about current colorectal cancer according to SNOMED ----
        #_get_snomed_codes()
        #df = _crc_snomed(df)
        #_compare_snomed_nlp(df)


        # ---- Manually exclude individuals with false positive matches ----
        exclusion_file = DATA_DIR / 'crc_exclude.csv'
        if exclusion_file.exists():
            df_excl = pd.read_csv(exclusion_file)
            subj_excl = df_excl.patient_id
            print('\nManually excluding some individuals with false positive NLP matches...')
            df.loc[df.patient_id.isin(subj_excl), 'crc_nlp'] = 0
            matches.loc[matches.patient_id.isin(subj_excl), 'exclusion_indicator'] = 1
            matches.loc[matches.patient_id.isin(subj_excl), 'exclusion_reason'] += 'manual exclusion;'
        
        # Retain pathology reports describing crc and drop report text
        dfpath = df.loc[df.crc_nlp==1].copy()  
        dfpath = dfpath.drop(labels='safe_report', axis=1) 

    # Extract earliest CRC date at this step?
    if min_date:
        idx = dfpath.groupby('patient_id')[datecol].idxmin()
        diag = dfpath.loc[idx]
    else:
        diag = dfpath
    
    # Retain only some columns
    if use_preprocessed_data:
        diag = diag[['patient_id', datecol, 'site_nlp', 'site_nlp_simple', 'row']].rename(columns={datecol:'diagnosis_date'}).copy()
    else:
        raise NotImplementedError
        diag = diag[['patient_id', datecol]].rename(columns={datecol:'diagnosis_date'}).copy()

    # Assign diagnosis source
    diag['diagnosis_source'] = 'pathology'

    return df, diag, matches
#endregion

# ==== CRC and its date from inpatient and outpatient data ====
#region
def crc_inpat_outpat(fit: pd.DataFrame, data_path: Path = DATA_DIR, min_date: bool = True):
    """Get CRC diagnosis and its date from inpatient and outpatient data"""
    print('\n---- Getting CRC diagnosis from inpat and outpat data ----')

    diag = pd.DataFrame()

    # ---- Diagnosis date from inpatient episodes ----

    ## Read data
    if PARQUET:
        df = pd.read_parquet(data_path / INPAT_DIAG_FILE)
    else:
        df = pd.read_csv(data_path / INPAT_DIAG_FILE)

    ## Retain patients with FIT
    df = df.loc[df.patient_id.isin(fit.patient_id)]

    ## Rename columns
    df = df.rename(columns={'diagnosis_date_time': 'diagnosis_date', 
                            'source_diagnosis_code_icd10': 'diagnosis_code_icd10'})

    ## Dates to datetime, plus drop clock time
    if not PARQUET:
        dq.check_date_format(df, datecols=['diagnosis_date'], formats=[DATETIME_FORMAT])
        df.diagnosis_date = pd.to_datetime(df.diagnosis_date, format=DATETIME_FORMAT)
    df.diagnosis_date = df.diagnosis_date.dt.normalize()

    ## Retain patients with CRC diag codes
    mask = df.diagnosis_code_icd10.astype(str).str.lower().str.contains('^c18|^c19|^c20', regex=True)
    df = df.loc[mask]

    ## Retain patients who have FIT
    df = df.loc[df.patient_id.isin(fit.patient_id)]

    ## Assign diagnosis source
    df['diagnosis_source'] = 'inpat'
    print('\nNumber of patients with inpatient C19-C20 diagnosis: {}'.format(df.patient_id.nunique()))

    # Get minimum date?
    if min_date:
        idx = df.groupby('patient_id')['diagnosis_date'].idxmin()
        tmp = df.loc[idx]
        tmp = tmp[['patient_id', 'diagnosis_date', 'diagnosis_source', 'diagnosis_code_icd10']]
    else:
        tmp = df[['patient_id', 'diagnosis_date', 'diagnosis_source', 'diagnosis_code_icd10']]
    diag = pd.concat(objs=[diag, tmp], axis=0).reset_index(drop=True)

    # ---- Diagnosis date from outpatient episodes ----

    ## Read data
    if PARQUET:
        df = pd.read_parquet(data_path / OUTPAT_DIAG_FILE)
    else:
        df = pd.read_csv(data_path / OUTPAT_DIAG_FILE)

    ## Retain patients with FIT
    df = df.loc[df.patient_id.isin(fit.patient_id)] 

    ## Rename columns
    df = df.rename(columns={'diagnosis_date_time': 'diagnosis_date', 
                            'source_diagnosis_code_icd10': 'diagnosis_code_icd10'})
    
    ## Dates to datetime plus drop clocktime
    if not PARQUET:
        dq.check_date_format(df, datecols=['diagnosis_date'], formats=[DATETIME_FORMAT])
        df.diagnosis_date = pd.to_datetime(df.diagnosis_date, format=DATETIME_FORMAT)
    df.diagnosis_date = df.diagnosis_date.dt.normalize()

    ## Retain patients with CRC diagnosis
    mask = df.diagnosis_code_icd10.astype(str).str.lower().str.contains('^c18|^c19|^c20', regex=True)
    df = df.loc[mask]

    ## Assign diagnosis source
    df['diagnosis_source'] = 'outpat'

    print('\nNumber of patients with outpatient C19-C20 diagnosis: {}'.format(df.patient_id.nunique()))

    # Get minimum date?
    if min_date:
        idx = df.groupby('patient_id')['diagnosis_date'].idxmin()
        tmp = df.loc[idx]
        tmp = tmp[['patient_id', 'diagnosis_date', 'diagnosis_source', 'diagnosis_code_icd10']]
    else:
        tmp = df[['patient_id', 'diagnosis_date', 'diagnosis_source', 'diagnosis_code_icd10']]
    diag = pd.concat(objs=[diag, tmp], axis=0).reset_index(drop=True)

    return diag
#endregion

# ==== CRC-relevant treatments and their dates ====
#region
def treatments(fit: pd.DataFrame, data_path: Path = DATA_DIR):
    """
    fit: DataFrame of individuals and earliest FIT values
    diag: DataFrame of single diagnosis date for each individual
    """
    print('\n---- Extracting treatments relevant for CRC ----')

    # Container
    tx = pd.DataFrame()

    # ---- Surgeries ----
    print('\n.... Getting procedure codes from surgeries')

    # Get procedure codes from inpatient and outpatient episodes
    if PARQUET:
        df1 = pd.read_parquet(data_path / INPAT_PROC_FILE)
    else:
        df1 = pd.read_csv(data_path / INPAT_PROC_FILE)
    df1['source_table'] = 'inpat'
    print('\nNumber of rows in inpatient table: {}'.format(df1.shape[0]))
    if not PARQUET:
        dq.check_date_format(df1, datecols=['procedure_date_time'], formats=[DATETIME_FORMAT])
        df1.procedure_date_time = pd.to_datetime(df1.procedure_date_time, format=DATETIME_FORMAT)
    df1 = df1.rename(columns={'procedure_date_time': 'start_date'})

    if PARQUET:
        outpat = pd.read_parquet(data_path / OUTPAT_FILE)
    else:
        outpat = pd.read_csv(data_path / OUTPAT_FILE)
    outpat = outpat[['patient_id', 'outp_attendance_id', 'attendance_date']].drop_duplicates()
    if PARQUET:
        df2 = pd.read_parquet(data_path / OUTPAT_PROC_FILE)
    else:
        df2 = pd.read_csv(data_path / OUTPAT_PROC_FILE)
    df2 = df2.merge(outpat, how='left')
    df2['source_table'] = 'outpat'
    print('\nNumber of rows in outpatient table: {}'.format(df2.shape[0]))
    if not PARQUET:
        dq.check_date_format(df2, datecols=['attendance_date'], formats=[DATETIME_FORMAT])
        df2.attendance_date = pd.to_datetime(df2.attendance_date, format=DATETIME_FORMAT)
    df2 = df2.rename(columns={'attendance_date': 'start_date'})

    proc = pd.concat(objs=[df1, df2], axis=0)
    print('\nNumber of rows in combined table: {}'.format(proc.shape[0]))

    # Get surgeries and polypectomy
    tdict = {'colonic stent': '^H214|^H243|^H244',
             'local excision': '^H402|^H412|^H34',
             'radical resection': '^H04|^H05|^H06|^H07|^H08|^H09|^H10|^H11|^H29|^H33|^X14',
             'polypectomy': '^H20|^H23'}
    df = pd.DataFrame()
    proc = proc.rename(columns={'procedure_date': 'treatment_date'})
    for key, val in tdict.items():
        tsub = proc.loc[proc.procedure_code_opcs.fillna('').str.upper().str.contains(val)].copy()
        tsub['event'] = key
        tsub = tsub[['patient_id', 'event', 'start_date']]
        print('Event:{}, count:{}'.format(key, tsub.shape[0]))
        df = pd.concat([df, tsub])

    # Store
    tx = pd.concat(objs=[tx, df], axis=0)


    # ---- Radiotherapy ----
    print('\n.... Getting radiotherapy data')
    if PARQUET:
        df = pd.read_parquet(data_path / RADIO_FILE)
    else:
        df = pd.read_csv(data_path / RADIO_FILE)

    if not PARQUET:
        dq.check_date_format(df, datecols=['start_date', 'end_date'], formats=[DATETIME_FORMAT])
        df.start_date = pd.to_datetime(df.start_date, format=DATETIME_FORMAT)
        df.end_date = pd.to_datetime(df.end_date, format=DATETIME_FORMAT)

    df = df[['patient_id', 'start_date']].drop_duplicates()
    df['event'] = 'radiotherapy'

    # Store
    tx = pd.concat(objs=[tx, df], axis=0)


    # ---- Chemotherapy ----
    print('\n.... Getting chemotherapy data')
    
    # Source 1
    if PARQUET:
        df = pd.read_parquet(data_path / CHEMO_ADMIN_FILE)
    else:
        df = pd.read_csv(data_path / CHEMO_ADMIN_FILE)
    print(df.shape[0], df.patient_id.nunique())

    if not PARQUET:
        dq.check_date_format(df, datecols=['date_time_administered', 'administration_stop_timestamp'],
                             formats=[DATETIME_FORMAT])
        df.date_time_administered = pd.to_datetime(df.date_time_administered, format=DATETIME_FORMAT)
        df.administration_stop_timestamp = pd.to_datetime(df.administration_stop_timestamp, format=DATETIME_FORMAT)

    df = df[['patient_id', 'date_time_administered']].rename(columns={'date_time_administered': 'start_date'})
    df['event'] = 'chemotherapy'
    df.start_date = df.start_date.dt.normalize()
    df = df.drop_duplicates()

    df1 = df.copy()

    # Source 2
    if PARQUET:
        df = pd.read_parquet(data_path / CHEMO_SUM_FILE)
    else:
        df = pd.read_csv(data_path / CHEMO_SUM_FILE)
    print(df.patient_id.nunique())
    print(df.columns)
    #if not PARQUET:   # in this parquet file, dates are not datetime format
    #dq.check_date_format(df, datecols=['start_date', 'end_date'], formats=[DATE_FORMAT])
    df.start_date = pd.to_datetime(df.start_date, format=DATE_FORMAT)
    df.end_date = pd.to_datetime(df.end_date, format=DATE_FORMAT)

    df = df[['patient_id', 'start_date']]  # .rename(columns={'start_date':'treatment_date'})
    df['event'] = 'chemotherapy'
    df2 = df.copy()

    # Dbl check
    df2.patient_id.isin(df1.patient_id).mean()
    df1.patient_id.isin(df2.patient_id).mean()

    df1min = df1.groupby(['patient_id']).start_date.min().rename('date1').reset_index()
    df2min = df2.groupby(['patient_id']).start_date.min().rename('date2').reset_index()
    dfmin = df1min.merge(df2min, how='inner')
    test = dfmin.date1 == dfmin.date2
    test.mean()
    dfmin['delta'] = dfmin.date1 - dfmin.date2
    dfmin.delta.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99])

    # Only use chemo admin data
    #df = pd.concat(objs=[df1, df2], axis=0)
    tx = pd.concat(objs=[tx, df1], axis=0)

    # ==== PROCESS ====

    # Drop duplicates
    print(tx.shape)
    tx = tx.drop_duplicates()
    tx = tx.reset_index(drop=True)
    print(tx.shape)

    # Only retain patients with FIT
    tx = tx.loc[tx.patient_id.isin(fit.patient_id)]
    print('\n{} of {} individuals have CRC-relevant treatment records'.format(tx.patient_id.nunique(), fit.patient_id.nunique()))
    
    return tx
#endregion

# ==== EARLIEST CRC DATE ====
#region
def earliest_crc_date(diag: pd.DataFrame, tx: pd.DataFrame, fit: pd.DataFrame,
                      run_path: Path, days_treatment_before_diag: int):

    print('\n---- Identifying earliest CRC date ----')

    # ==== Get earliest crc date among path, inpat, outpat ====
    diag = diag.reset_index(drop=True)
    idx = diag.groupby('patient_id')['diagnosis_date'].idxmin()
    diagmin = diag.loc[idx]
    if diagmin.patient_id.nunique() != diagmin.shape[0]:
        raise ValueError('More than one crc date per individual')

    s = diagmin.groupby('diagnosis_source')['patient_id'].nunique()
    print('\nSource of earliest diagnosis date:')
    print(s)


    # ---- If crc-like treatments occur within 180 days of crc date, use their date as crc date ----

    # Check if any treatments occur within 180 days before earliest diag date
    tmp = tx[['patient_id', 'event', 'start_date']].rename(columns={'start_date': 'treatment_date', 
                                                                    'event': 'treatment'})
    tmp = diagmin.merge(tmp, how='left', on=['patient_id'])
    tmp['days_treatment_to_diag'] = (tmp.diagnosis_date - tmp.treatment_date).dt.days
    mask = (tmp.days_treatment_to_diag <= days_treatment_before_diag) & (tmp.days_treatment_to_diag > 0)
    tmp = tmp.loc[mask]
    print(tmp.days_treatment_to_diag.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))

    # Treatment date as diag date
    tmp['diagnosis_date_path_or_code'] = tmp.diagnosis_date
    tmp['diagnosis_source_path_or_code'] = tmp.diagnosis_source
    tmp.diagnosis_date = tmp.treatment_date
    tmp.diagnosis_source = 'start_of_treatment'
    tmp = tmp[['patient_id', 'diagnosis_date', 'diagnosis_source', 'treatment', 'days_treatment_to_diag']]
    tmp = tmp.reset_index(drop=True)

    # Retain earliest
    print(tmp.shape)
    tmp = tmp.reset_index(drop=True)
    idx = tmp.groupby('patient_id')['diagnosis_date'].idxmin()
    tmp = tmp.loc[idx]
    print(tmp.shape)
    print(tmp.head())

    # Replace in diagmin
    shape0 = diagmin.shape[0]
    print(diagmin.shape)
    diagmin0 = diagmin.loc[~diagmin.patient_id.isin(tmp.patient_id)]
    diagmin1 = diagmin.loc[diagmin.patient_id.isin(tmp.patient_id)]
    diagmin1 = diagmin1.rename(columns={'diagnosis_source': 'diagnosis_source_path_or_code', 'diagnosis_date': 'diagnosis_date_path_or_code'})
    tmp = tmp.merge(diagmin1, how='left')
    diagmin = pd.concat(objs=[diagmin0, tmp], axis=0)
    assert shape0 == diagmin.shape[0]

    # Summarise again
    print(diagmin.patient_id.nunique(), diagmin.shape[0])
    s = diagmin.groupby('diagnosis_source')['patient_id'].nunique()
    print('\nSource of earliest diagnosis date, after updating using tx:')
    print(s)

    # Dbl heck that each individual has one date
    if diagmin.patient_id.nunique() != diagmin.shape[0]:
        raise ValueError('More than one crc date per individual')

    # ---- See patterns of path, inpat, and outpat crc dates ----

    # Difference in days between pathology report and inpat diagnosis code for CRC
    min_date = False
    if min_date:
        dp = diag.pivot(index='patient_id', columns='diagnosis_source', values='diagnosis_date')
        dp['delta'] = (dp.pathology - dp.inpat).dt.days
        s = dp.delta.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        print('\nDif in days between date of path and date of inpat: \n{}'.format(s))

        test = ~dp.drop(labels='delta', axis=1).isna()
        test['inp_and_path'] = test.inpat & test.pathology
        test['inp_only'] = test.inpat & ~test.pathology
        test['path_only'] = ~test.inpat & test.pathology
        s = test[['inp_and_path', 'inp_only', 'path_only']].mean(axis=0)
        print('\nProportion of patients with pathology and inpatient diagnosis: \n{}'.format(s))

        dp = dp.sort_values(by=['delta', 'pathology', 'inpat', 'outpat'], ascending=[True, True, True, True])
        print(dp)

    # ---- More summaries ----

    # Count patients for whom FIT value occurred after CRC diag
    diagmin = diagmin.merge(fit, how='left', on='patient_id')
    diagmin['crc_before_fit'] = 0
    mask = diagmin.fit_date > diagmin.diagnosis_date
    diagmin.loc[mask, 'crc_before_fit'] = 1
    print('Number of patients with CRC and earliest FIT value: {}'.format(diagmin.patient_id.nunique()))
    print('Number of patients with CRC before earliest FIT value: {}'.format(diagmin.crc_before_fit.sum()))

    # Explore time from FIT to diagnosis
    #  Note that sometimes the diagnosis occurs very close to FIT date
    diagmin['days_fit_to_diag'] = diagmin.diagnosis_date - diagmin.fit_date
    diagmin['days_fit_to_diag'] = diagmin.days_fit_to_diag.dt.days + diagmin.days_fit_to_diag.dt.seconds / (60 * 60 * 24)
    s = diagmin.days_fit_to_diag.describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    print('\nDays from FIT to CRC:')
    print(s)
    plt.hist(diagmin.days_fit_to_diag, bins=20)
    plt.xlabel('Days from FIT to CRC')
    plt.ylabel('Number of patients')
    plt.savefig(run_path / PLOT_DAYS_FIT_CRC, dpi=150, bbox_inches='tight')
    plt.close()

    test_365 = (diagmin.days_fit_to_diag >= 0) & (diagmin.days_fit_to_diag <= 365)
    test_180 = (diagmin.days_fit_to_diag >= 0) & (diagmin.days_fit_to_diag <= 180)
    print('Diagnosis date is within [0, 365] days from FIT date in {} ({}%) of cases'.format(test_365.sum(), test_365.mean() * 100))
    print('Diagnosis date is within [0, 180] days from FIT date in {} ({}%) of cases'.format(test_180.sum(), test_180.mean() * 100))

    # Number of patients with time from FIT to CRC
    #  E.g. X patients had records of CRC more than 2 years after FIT
    thr = [0, 1, 5, 14, 30, 180, 365, 2 * 365, 3 * 365, 4 * 365]
    check = pd.DataFrame()
    for t in thr:
        dsub = diagmin.loc[diagmin['days_fit_to_diag'] > t]
        c = pd.DataFrame([[t, dsub.patient_id.nunique()]])
        check = pd.concat(objs=[check, c], axis=0)
    check.columns = ['days_nodiag', 'n_patient']
    print('\nNumber of patients with days from FIT to CRC greater than threshold:')
    print(diagmin.patient_id.nunique())
    print(check)

    return diagmin
#endregion