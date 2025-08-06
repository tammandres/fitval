"""Get data items used for modelling and apply inclusion criteria
Data items will later be converted to a matrix of predictor variables
"""
from constants import DATA_DIR
from dataprep.crc import crc_and_tx
from dataprep.demographics import demographics
from dataprep.files import OutputFiles
from dataprep.fit import prepare_fit
from dataprep.imaging import imaging_events
from dataprep.stage import stage
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import regex as re

from dataprep.bloods import bloods
from dataprep.bmi import bmi
from dataprep.codes import codes
#from textmining.symptoms import symptoms

# Output data files
f_out = OutputFiles()
FIT_OUT = f_out.fit
#CLIND_OUT = f_out.clind
SYM_OUT = f_out.sym
DEMO_OUT = f_out.demo
DIAGMIN_OUT = f_out.diagmin
DIAG_OUT = f_out.diag
EVENTS_OUT = f_out.events
BLOODS_OUT = f_out.bloods
#BLOODS_HILO_OUT = f_out.bloods_hilo
DIAG_CODES_OUT = f_out.diag_codes
PROC_CODES_OUT = f_out.proc_codes
PRES_CODES_OUT = f_out.pres_codes
BMI_OUT = f_out.bmi


@dataclass
class CoreData:
    """Core data: earliest FIT values and associated data that defines a cohort of patients"""
    fit: pd.DataFrame   # Earliest FIT values for each individual, and clinical details
    #clind: pd.DataFrame  # Clinical details for FIT values
    sym: pd.DataFrame  # Clinical symptoms extracted from clinical details
    demo: pd.DataFrame  # Demographics
    diagmin: pd.DataFrame  # Earliest CRC date and source
    diag: pd.DataFrame  # ALL CRC dates and sources
    events: pd.DataFrame  # Event logs


@dataclass
class AdditionalData:
    """Additional data items needed for prediction"""
    bloods: pd.DataFrame  # Blood test results
    #bloods_hilo: pd.DataFrame  # High-low indicators for certain bloods
    diag_codes: pd.DataFrame  # Diagnosis codes (inpat and outpat)
    proc_codes: pd.DataFrame  # Procedure codes (inpat and outpat)
    pres_codes: pd.DataFrame  # Prescription codes
    bmi: pd.DataFrame  # body mass index


def dates_to_datetime(dfs):
    """dfs: list of dataframes"""
    in_pattern = 'date|dts'
    ex_pattern = 'indicator'
    for df in dfs:
        for col in df.columns:
            if re.search(in_pattern, col) and not re.search(ex_pattern, col):
                print('Converting column {} to datetime'.format(col))
                df[col] = pd.to_datetime(df[col])


def load_coredata(run_path: Path):
    print('\n==== LOADING CORE DATA ====\n')

    fit = pd.read_csv(run_path / FIT_OUT)
    
    sym_path = run_path / SYM_OUT
    if sym_path.exists():
        sym = pd.read_csv(run_path / SYM_OUT)
    else:
        sym = pd.DataFrame()

    demo = pd.read_csv(run_path / DEMO_OUT)
    diagmin = pd.read_csv(run_path / DIAGMIN_OUT, dtype={'T': str, 'N': str, 'M': str})
    diag = pd.read_csv(run_path / DIAG_OUT)

    events_path = run_path / EVENTS_OUT
    if events_path.exists():
        events = pd.read_csv(run_path / EVENTS_OUT)
    else:
        events = pd.DataFrame()

    dfs = [fit, sym, demo, diagmin, diag, events]
    dates_to_datetime(dfs)

    return CoreData(fit=fit, sym=sym, demo=demo, diagmin=diagmin, diag=diag, events=events)


def save_coredata(cdata: CoreData, save_path: Path):
    print('\n==== SAVING CORE DATA ====\n')
    field_file_map = dict(fit=FIT_OUT, sym=SYM_OUT, demo=DEMO_OUT, 
                          diagmin=DIAGMIN_OUT, diag=DIAG_OUT, events=EVENTS_OUT)
    
    for field, file in field_file_map.items():
        df = getattr(cdata, field)
        if df.shape[0] > 0:
            print('Saving {}'.format(field))
            df.to_csv(save_path / file, index=False)
    print('Save complete.')


def load_additional_data(run_path: Path):
    print('\n==== LOADING ADDITIONAL DATA ====\n')
    
    bloods_path = run_path / BLOODS_OUT
    diag_path = run_path / DIAG_CODES_OUT
    proc_path = run_path / PROC_CODES_OUT
    pres_path = run_path / PRES_CODES_OUT
    bmi_path = run_path / BMI_OUT
    
    bloods = pd.read_csv(bloods_path) if bloods_path.exists() else pd.DataFrame()
    diag_codes = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    proc_codes = pd.read_csv(proc_path) if proc_path.exists() else pd.DataFrame()
    pres_codes = pd.read_csv(pres_path) if pres_path.exists() else pd.DataFrame()
    bmi = pd.read_csv(bmi_path) if bmi_path.exists() else pd.DataFrame()

    dfs = [bloods, diag_codes, proc_codes, pres_codes, bmi]
    dates_to_datetime(dfs)

    data = AdditionalData(bloods=bloods, diag_codes=diag_codes, proc_codes=proc_codes, pres_codes=pres_codes, bmi=bmi)

    return data


def save_additional_data(adata: CoreData, save_path: Path):
    print('\n==== SAVING ADDITIONAL DATA ====\n')
    field_file_map = dict(bloods=BLOODS_OUT, diag_codes=DIAG_CODES_OUT, 
                          proc_codes=PROC_CODES_OUT, pres_codes=PRES_CODES_OUT, bmi=BMI_OUT)
    
    for field, file in field_file_map.items():
        df = getattr(adata, field)
        if df.shape[0] > 0:
            print('Saving {}'.format(field))
            df.to_csv(save_path / file, index=False)
    print('Save complete.')


def coredata(run_path: Path, data_path: Path = DATA_DIR, save_data: bool = True, 
             testmode: bool = False, 
             days_treatment_before_diag: int = 180,
             days_stage_after_diag: int = 180, gp_only: bool = False, infoflex: bool = False):
    """Get core data: 
    earliest FIT values for each patient, 
    and associated demographics, CRC date, TNM stage, event logs.
    These data sources will later be filtered according to study inclusion and exclusion criteria.

    Args:
        run_path: where to save data
        data_path: where to load data
    """

    # Fit
    fit, nfit = prepare_fit(run_path, data_path, gp_only=gp_only, save_data=save_data)

    # Demographics
    demo = demographics(fit, run_path, data_path, save_data=save_data)

    # CRC date, and treatments relevant for CRC
    diagmin, diag, __, __, tx = crc_and_tx(fit, run_path, data_path=data_path, save_data=save_data, testmode=testmode,
                                           days_treatment_before_diag=days_treatment_before_diag, infoflex=infoflex)

    # Add max T-stage to diagmin; also get events logs for any T-N-M values
    diagmin, events_stage = stage(run_path, fit, diagmin, save_data=save_data, testmode=testmode, 
                                  days_stage_after_diag=days_stage_after_diag)

    # Imaging events
    img = imaging_events(fit)

    # Event logs
    events = event_log(run_path, demo, tx, img, fit, diagmin, events_stage, diag, save_data=save_data)
    
    # Process clinical symptoms
    sym_cols = [c for c in fit.columns if c.startswith('sym')]
    dfsym = pd.melt(fit, id_vars=['patient_id'], value_vars=sym_cols, var_name='category')
    dfsym = dfsym.loc[dfsym.value==1]
    dfsym.category = dfsym.category.str.replace('symptom_', '')
    if save_data:
        dfsym.to_csv(run_path / SYM_OUT, index=False)

    return CoreData(fit=fit, sym=dfsym, demo=demo, diagmin=diagmin, diag=diag, events=events)


def additional_data(run_path: Path, fit: pd.DataFrame, data_path: Path = DATA_DIR,
                    save_data: bool = True, testmode: bool = False, npat_bloods: int = 100,
                    days_before_fit_blood: int = 365, days_after_fit_blood: int = 30):
    """The purpose of applying days_before_fit_blood, days_after_fit_blood, and npat_bloods
    here is to reduce the size of blood tests table saved to disk; 
    later, when applying inclusion criteria, the table will be filtered more specifically
    """

    # Bloods
    b, __ = bloods(run_path, fit, data_path, save_data=save_data, testmode=testmode, npat=npat_bloods,
                   days_before_fit_blood=days_before_fit_blood, days_after_fit_blood=days_after_fit_blood)

    # Diagnosis, procedure, presciription codes
    cdiag, cproc, cpres = codes(run_path, fit, data_path, save_data=save_data, testmode=testmode)
    
    # Body mass index
    df_bmi = bmi(run_path, fit, data_path, save_data=save_data, testmode=testmode)
    #df_bmi = pd.DataFrame()
    return AdditionalData(bloods=b, diag_codes=cdiag, proc_codes=cproc, pres_codes=cpres, bmi=df_bmi)


def event_log(run_path: Path, 
              demo: pd.DataFrame, tx: pd.DataFrame, img: pd.DataFrame, 
              fit: pd.DataFrame, diagmin: pd.DataFrame, events_stage: pd.DataFrame, 
              diag: pd.DataFrame,
              save_data=True):
    """Generate event logs: dataframe with main columns ['patient_id', 'event', 'start_date']"""
    print('\n==== GENERATING EVENT LOGS ====\n')

    dfe = pd.DataFrame()

    # Last alive date
    #df = demo[['patient_id', 'last_alive_date']].rename(columns={'last_alive_date':'start_date'}).dropna()
    #df['event'] = 'last alive'
    #dfe = pd.concat(objs=[dfe, df], axis=0)

    # Death date
    df = demo[['patient_id', 'death_date']].rename(columns={'death_date':'start_date'}).dropna()
    df['event'] = 'death'
    dfe = pd.concat(objs=[dfe, df], axis=0)

    # Treatments
    #df = tx[['patient_id', 'start_date', 'event']].copy()
    dfe = pd.concat(objs=[dfe, tx], axis=0)

    # Imaging data
    dfe = pd.concat(objs=[dfe, img], axis=0)

    # FIT data in event log 
    df = fit[['patient_id', 'fit_date']].rename(columns={'fit_date':'start_date'}).drop_duplicates()
    df['event'] = 'FIT test'
    dfe = pd.concat(objs=[dfe, df], axis=0)

    # Diagnosis data in event log 
    df = diagmin[['patient_id', 'diagnosis_date', 'diagnosis_source']].rename(columns={'diagnosis_date':'start_date'})
    df = df.drop_duplicates()
    df['event'] = 'diagnosis'
    dfe = pd.concat(objs=[dfe, df], axis=0)

    # Staging data
    dfe = pd.concat(objs=[dfe, events_stage], axis=0)

    # Add other cancer diagnosis events
    df = diag[['patient_id', 'diagnosis_date', 'diagnosis_source']].rename(columns={'diagnosis_date':'start_date'})
    df = df.drop_duplicates()
    df['event'] = 'crc_' + df.diagnosis_source
    dfe = pd.concat(objs=[dfe, df], axis=0)

    # Add diagnosis date, and compute days to diag 
    dfe = dfe.merge(diagmin[['patient_id', 'diagnosis_date']], how='left', on='patient_id')
    dfe = dfe.merge(fit[['patient_id', 'fit_date']], how='left', on='patient_id')
    dfe['days_diag_to_event'] = (dfe.start_date - dfe.diagnosis_date).dt.days
    dfe['days_fit_to_event'] = (dfe.start_date - dfe.fit_date).dt.days

    # Add indicator for treatment events
    dfe['treatment'] = 0
    dfe.loc[dfe.event.isin(['local excision', 'radical resection', 'chemotherapy', 'radiotherapy', 
                            'colonic stent']), 'treatment'] = 1

    # Remove clock time
    dfe.start_date = dfe.start_date.dt.normalize()

    # Little summary
    print('All events included: {}'.format(dfe.event.unique()))
    print('Treatment events: {}'.format(dfe.loc[dfe.treatment==1].event.unique()))

    if save_data:
        dfe.to_csv(run_path / EVENTS_OUT, index=False)

    return dfe
