"""Get diagnosis, procedure, and presciption codes associated with FIT values"""
from constants import DATA_DIR, DATETIME_FORMAT, DATE_FORMAT, PARQUET
from dataprep import dq
from dataprep.files import InputFiles, OutputFiles, DQFiles
from pathlib import Path
import numpy as np
import pandas as pd

f_in = InputFiles()
f_out = OutputFiles()
f_dq = DQFiles()

# Input files
INPAT_DIAG_FILE = f_in.inpat_diag
OUTPAT_DIAG_FILE = f_in.outpat_diag
INPAT_PROC_FILE = f_in.inpat_proc
OUTPAT_PROC_FILE = f_in.outpat_proc
PRESCRIPTION_FILE = f_in.prescribing
OUTPAT_FILE = f_in.outpat

# Output files
DIAG_CODES_OUT = f_out.diag_codes
PROC_CODES_OUT = f_out.proc_codes
PRES_CODES_OUT = f_out.pres_codes
CODE_COUNTS = f_dq.code_counts


def codes(run_path: Path, fit: pd.DataFrame, data_path: Path = DATA_DIR, 
          save_data: bool = True, testmode: bool = False):
    """Extract diagnosis, procedure and prescription codes
    within range [-days_before_fit_event, days_after_fit_event] of earliest FIT
    and include codes available for at least npat patients"""
    print('\n==== EXTRACTING DIAGNOSIS, PROCEDURE, PRESCRIPTION CODES ====')
    
    # Diagnosis codes
    df_diag = diag_codes(run_path, fit=fit, data_path=data_path, save_data=save_data, testmode=testmode)

    # Procedure codes
    df_proc = proc_codes(run_path, fit=fit, data_path=data_path, save_data=save_data, testmode=testmode)

    # Prescription codes
    df_pres = pres_codes(run_path, fit=fit, data_path=data_path, save_data=save_data, testmode=testmode)

    return df_diag, df_proc, df_pres


def diag_codes(run_path: Path, fit: pd.DataFrame, data_path: Path = DATA_DIR,
               save_data: bool = True, testmode: bool = False):
    """Get all diagnosis codes for at least npat patients and
    within range of [-days_before_fit_event, days_after_fit_event] from earliest FIT test"""

    nrows = 5000 if testmode else None

    # Diagnosis codes from inpatient and outpatient episodes
    if PARQUET:
        df1 = pd.read_parquet(data_path / INPAT_DIAG_FILE)
    else:
        df1 = pd.read_csv(data_path / INPAT_DIAG_FILE, nrows=nrows)
    df1['source_table'] = 'inpat'
    df1 = df1.rename(columns={'diagnosis_date_time': 'diagnosis_date'})
    print('Number of rows in inpatient table: {}'.format(df1.shape[0]))

    if not PARQUET:
        dq.check_date_format(df1, datecols=['diagnosis_date'], formats=[DATETIME_FORMAT])
        df1.diagnosis_date = pd.to_datetime(df1.diagnosis_date, format=DATETIME_FORMAT)
        
    if PARQUET:
        df2 = pd.read_parquet(data_path / OUTPAT_DIAG_FILE)
    else:
        df2 = pd.read_csv(data_path / OUTPAT_DIAG_FILE, nrows=nrows)
    df2 = df2.rename(columns={'diagnosis_date_time': 'diagnosis_date'})
    df2['source_table'] = 'outpat'
    print('Number of rows in outpatient table: {}'.format(df2.shape[0]))

    if not PARQUET:
        dq.check_date_format(df2, datecols=['diagnosis_date'], formats=[DATETIME_FORMAT])
        df2.diagnosis_date = pd.to_datetime(df2.diagnosis_date, format=DATETIME_FORMAT)

    df = pd.concat(objs=[df1, df2], axis=0)
    print('Number of rows in combined table: {}'.format(df.shape[0]))

    if testmode:
        df = df.sample(5000, random_state=42)

    df = df.rename(columns={'source_diagnosis_code_icd10': 'diagnosis_code_icd10'})
    
    # Exclude CRC diagnosis codes
    mask = df.diagnosis_code_icd10.astype(str).str.lower().str.contains('^c18|^c19|^c20', regex=True)
    df = df.loc[~mask]
    print('Number of diagnosis codes (excluding C18-C20): {}'.format(df.shape[0]))

    # Retain patients with fit
    df = df.loc[df.patient_id.isin(fit.patient_id)]

    # Simplify 
    df = df[['patient_id', 'diagnosis_code_icd10', 'diagnosis_date']]
    df = df.rename(columns={'diagnosis_code_icd10':'event', 'diagnosis_date':'event_date'})
    df['event_type'] = 'diagnosis_code'
    df.event_date = df.event_date.dt.normalize()

    if save_data:
        df.to_csv(run_path / DIAG_CODES_OUT, index=False)

    return df


def proc_codes(run_path: Path, fit: pd.DataFrame, data_path: Path = DATA_DIR, 
               save_data: bool = True, testmode: bool = False):

    """Get all procedure codes for at least npat patients and
    within range of [-days_before_fit_event, days_after_fit_event] from earliest FIT test"""
    nrows = 5000 if testmode else None

    # Procedure codes from inpatient and outpatient episodes
    if PARQUET:
        df1 = pd.read_parquet(data_path / INPAT_PROC_FILE)
    else:
        df1 = pd.read_csv(data_path / INPAT_PROC_FILE, nrows=nrows)
    df1['source_table'] = 'inpat'
    print('\nNumber of rows in inpatient table: {}'.format(df1.shape[0]))
    if not PARQUET:
        dq.check_date_format(df1, datecols=['procedure_date_time'], formats=[DATETIME_FORMAT])
        df1.procedure_date_time = pd.to_datetime(df1.procedure_date_time, format=DATETIME_FORMAT)
    df1 = df1.rename(columns={'procedure_date_time': 'procedure_date'})

    if PARQUET:
        outpat = pd.read_parquet(data_path / OUTPAT_FILE)
    else:
        outpat = pd.read_csv(data_path / OUTPAT_FILE, nrows=nrows)
    outpat = outpat[['patient_id', 'outp_attendance_id', 'attendance_date']].drop_duplicates()

    if PARQUET:
        df2 = pd.read_parquet(data_path / OUTPAT_PROC_FILE)
    else:
        df2 = pd.read_csv(data_path / OUTPAT_PROC_FILE, nrows=nrows)

    df2 = df2.merge(outpat, how='left')
    df2['source_table'] = 'outpat'
    print('\nNumber of rows in outpatient table: {}'.format(df2.shape[0]))
    if not PARQUET:
        dq.check_date_format(df2, datecols=['attendance_date'], formats=[DATETIME_FORMAT])
        df2.attendance_date = pd.to_datetime(df2.attendance_date, format=DATETIME_FORMAT)
    df2 = df2.rename(columns={'attendance_date': 'procedure_date'})

    df = pd.concat(objs=[df1, df2], axis=0)
    print('Number of rows in combined table: {}'.format(df.shape[0]))

    # Simplify 
    df = df[['patient_id', 'procedure_code_opcs', 'procedure_date']]
    df = df.rename(columns={'procedure_code_opcs':'event', 'procedure_date':'event_date'})
    df['event_type'] = 'procedure_code'
    df.event_date = df.event_date.dt.normalize()

    # Retain patients with fit
    df = df.loc[df.patient_id.isin(fit.patient_id)]

    # Double check if any CRC procedures occur in the data (these shouldn't appear before CRC diagnosis)
    tdict = {'colonic stent': '^H214|^H243|^H244',
             'local excision': '^H402|^H412|^H34',
             'radical resection': '^H04|^H05|^H06|^H07|^H08|^H09|^H10|^H11|^H29|^H33|^X14',
             'polypectomy':'^H20|^H23'}
    for key, pat in tdict.items():
        mask = df.event.astype(str).str.upper().str.contains(pat, regex=True)
        df = df.loc[~mask]
        print('Number of procedure codes for {} in data: {}'.format(key, mask.sum()))

    if save_data:
        df.to_csv(run_path / PROC_CODES_OUT, index=False)

    return df


def pres_codes(run_path: Path, fit: pd.DataFrame, data_path: Path = DATA_DIR, 
               save_data: bool = True, testmode: bool = False):

    """Get all procedure codes for at least npat patients and
    within range of [-days_before_fit_event, days_after_fit_event] from earliest FIT test"""

    nrows = 5000 if testmode else None

    # Prescription codes
    if PARQUET:
        df = pd.read_parquet(data_path / PRESCRIPTION_FILE)
    else:
        df = pd.read_csv(data_path / PRESCRIPTION_FILE, nrows=nrows)
    print('Number of rows in table: {}'.format(df.shape[0]))
    if not PARQUET:
        dq.check_date_format(df, datecols=['prescription_date_time'], formats=[DATETIME_FORMAT])
        df.prescription_date_time = pd.to_datetime(df.prescription_date_time, format=DATETIME_FORMAT)
    df.prescription_date_time = df.prescription_date_time.dt.normalize()

    # Retain patients with fit
    df = df.loc[df.patient_id.isin(fit.patient_id)]

    # Simplify 
    df = df[['patient_id', 'drug_name', 'prescription_date_time']]
    df = df.rename(columns={'drug_name':'event', 'prescription_date_time':'event_date'})
    df['event_type'] = 'medication'

    if save_data:
        df.to_csv(run_path / PRES_CODES_OUT, index=False)

    return df
    

def filter_codes(df, fit, diagmin, days_before_fit_event, days_after_fit_event, npat, rmlast=False):

    # Time from procedure code to FIT test in days
    if 'fit_date' in df.columns:
        df = df.drop(labels=['fit_date'], axis=1)
    df = df.merge(fit[['patient_id', 'fit_date']], how='inner')
    df['days_fit_to_event'] = df.event_date - df.fit_date
    df.days_fit_to_event = df.days_fit_to_event.dt.days

    # Retain observations close to FIT date
    mask = (df.days_fit_to_event < days_after_fit_event) & (df.days_fit_to_event > -days_before_fit_event)
    df = df.loc[mask]
    print('Number of observations within {} days before and {} days after FIT: {}'.format(days_before_fit_event, days_after_fit_event, df.shape[0]))

    # Ensure all codes occurred before CRC date if CRC was present
    df = df.merge(diagmin[['patient_id', 'diagnosis_date']].rename(columns={'diagnosis_date':'crc_date'}), how='left')
    mask = df.event_date >= df.crc_date 
    print('Number of observations occurring after or at CRC date: {}'.format(mask.sum()))
    df = df.loc[~mask]
    print('Number of observations left after dropping results after or at CRC date: {}'.format(df.shape[0]))

    # Drop last digit from code 
    if rmlast:
        print('\nRetaining first three characters of codes...')
        print(df.event.nunique())
        df.event = df.event.str.replace(r'\W', '', regex=True).str[:3]
        print(df.event.nunique())

    # Explore codes available for less than npat patients
    if npat is not None:
        s = df.groupby(['event'])['patient_id'].nunique().rename('nsub').reset_index()
        mask = s.nsub < npat
        print('Number of codes available for less than {} patients: {}'.format(npat, mask.sum()))
        print('\nCodes available for less than {} patients: \n{}'.format(npat, s.loc[mask]))
        print('\nCodes available for {} patients or more: \n{}'.format(npat, s.loc[~mask]))

    # Drop codes available for less than npat patients
    if npat is not None:
        s = s.loc[~mask]
        df = df.merge(s, how='inner')
        print('Number of observations left in codes table after dropping rare codes: {}'.format(df.shape[0]))

    return df
