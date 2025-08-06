"""Apply inclusion criteria to data, and generate a summary table"""
import numpy as np
import pandas as pd
from constants import DATACUT
from dataprep.coredata import load_coredata, load_additional_data, save_coredata, save_additional_data
from dataprep.files import OutputFiles
from dataprep.bloods import filter_bloods
from dataprep.codes import filter_codes
from dataprep.bmi import filter_bmi
from dataclasses import fields
from pathlib import Path

# Output file - to save patient counts after applying inclusion criteria
f_out = OutputFiles()
INCLUSION_FILE = f_out.inclusion


def inclusion(data_path: Path, run_path: Path, days_followup: int = 180, datacut=DATACUT, 
              sym_only: bool = False, save_data: bool = True, 
              ex_treatment_before_fit: bool = False, ex_treatment_after_fit: bool = False,
              days_before_fit_blood: int = 60, days_after_fit_blood: int = 30,
              days_before_fit_event: int = 365, days_after_fit_event: int = 0,
              npat_blood: int = 100, npat_code: int = 100,
              coreblood: bool = True, coreblood_colofit: bool = False, buffer: str = 'none',
              ppat_blood: float = None, clinical_history: bool = False, crc_before_fu: bool = None):
    """Apply inclusion and exclusion criteria
    1) Which patient_ids are included
    2) Date range of data points that are included
    """
    if buffer not in ['none', 'pre-adoption', 'post-adoption', 'fit-comment']:
        raise ValueError("'buffer' must be in ['none', 'pre-adoption', 'post-adoption', 'fit-comment']")
    if npat_blood is not None and ppat_blood is not None:
        raise ValueError("npat_blood and ppat_blood must not both be defined; one must be set to None")
    if crc_before_fu:
        raise NotImplementedError

    # Load data
    cdata = load_coredata(data_path)
    adata = load_additional_data(data_path)

    # To keep track of inclusion-exclusion steps
    flow = pd.DataFrame()
    txt = ""

    # Initial FIT value counts
    txt += 'FIT value'
    flow = _add_counts(txt, flow, cdata)

    # Exclude FITs not ordered by GP
    subj = cdata.fit.loc[cdata.fit.gp==1].patient_id.unique()
    cdata = _include_subj(cdata, subj)
    adata = _include_subj(adata, subj)
    txt += ', GP FIT'
    flow = _add_counts(txt, flow, cdata)
    
    # Exclude patients less than 18 years old at FIT 
    df = cdata.demo.loc[cdata.demo.age_at_fit >= 18]
    subj = df.patient_id.unique()
    cdata = _include_subj(cdata, subj)
    adata = _include_subj(adata, subj)
    txt += ', age >= 18'
    flow = _add_counts(txt, flow, cdata)

    # Ensure each individual has 'days_followup' number of days after FIT test
    # This means that there must be enough days between FIT test date and datacut date
    print('Datacut date: {}'.format(datacut.iloc[0]))
    cdata.fit['fu_date'] = cdata.fit.fit_date + pd.to_timedelta(days_followup, unit='days')  
    cdata.fit['datacut'] = datacut.iloc[0]
    cdata.fit['has_fu'] = 0
    cdata.fit.loc[cdata.fit.fu_date <= cdata.fit.datacut, 'has_fu'] = 1  

    msg = '{} ({:.2f}%) of {} patients have required follow-up after FIT'
    print(msg.format(cdata.fit.has_fu.sum(), cdata.fit.has_fu.mean()*100, cdata.fit.patient_id.nunique()))

    cdata.fit = cdata.fit.loc[cdata.fit.has_fu==1]
    subj = cdata.fit.patient_id.unique()

    cdata = _include_subj(cdata, subj)
    adata = _include_subj(adata, subj)
    txt += ', ' + str(days_followup) + '-day follow-up'
    flow = _add_counts(txt, flow, cdata)

    # Exclude CRC patients when CRC occurred before earliest FIT value
    subj_ex = cdata.diagmin.loc[cdata.diagmin.crc_before_fit==1].patient_id.unique()
    print('\nIndividuals with CRC before FIT')
    print(cdata.diag.loc[cdata.diag.patient_id.isin(subj_ex)])

    subj = np.setdiff1d(cdata.fit.patient_id, subj_ex)
    cdata = _include_subj(cdata, subj)
    adata = _include_subj(adata, subj)
    txt += ', CRC after FIT'
    flow = _add_counts(txt, flow, cdata)

    # If CRC occurred after follow-up, place these individuals in no-CRC group
    # E.g. if follow-up is 180 days after FIT and someone had cancer at day 190,  they are counted as having no cancer
    # Note that data has already been filtered, so that each individual has required follow up
    cdata.diagmin = cdata.diagmin
    print(cdata.diagmin.shape)
    cdata.diagmin = cdata.diagmin.merge(cdata.fit[['patient_id', 'fu_date']], how='left')
    print(cdata.diagmin.shape)

    mask = (cdata.diagmin.diagnosis_date <= cdata.diagmin.fu_date) & (cdata.diagmin.diagnosis_date >= cdata.diagmin.fit_date)
    cdata.diagmin['crc_within_fu'] = 0
    cdata.diagmin.loc[mask, 'crc_within_fu'] = 1
    cdata.diagmin = cdata.diagmin.loc[mask]
    cdata.diag = cdata.diag.loc[cdata.diag.patient_id.isin(cdata.diagmin.patient_id)]

    txt += ', CRC before follow-up date'
    flow = _add_counts(txt, flow, cdata)

    # Exclude patients without FIT-relevant symptoms?
    if sym_only:

        # Retain individuals with NICE symptoms
        syms = ['abdopain', 'abdomass', 'ida', 'anaemia', 'tarry', 'bloodsympt', 'diarr',
                'bowelhabit', 'constipation', 'rectalpain', 'rectalmass', 'rectalulcer', 'wl']
        dfsym = cdata.sym
        nicesym = dfsym.loc[dfsym.category.isin(syms)]
        subj = nicesym.patient_id.unique()
        cdata = _include_subj(cdata, subj)
        adata = _include_subj(adata, subj)

        txt += ', symptoms'
        flow = _add_counts(txt, flow, cdata)

    # Exclude patients with CRC-like treatments before FIT?
    if ex_treatment_before_fit:

        e = ['colonic stent', 'local excision', 'radical resection', 'polypectomy', 'radiotherapy', 'chemotherapy']
        dfsub = cdata.events.loc[cdata.events.event.isin(e)]
        mask = dfsub.start_date <= dfsub.fit_date
        subj_ex = dfsub.loc[mask].patient_id.unique()
        print('{} individuals have some CRC-like treatments recorded before FIT date'.format(len(subj_ex)))
        subj = np.setdiff1d(cdata.fit.patient_id, subj_ex)

        cdata = _include_subj(cdata, subj)
        adata = _include_subj(adata, subj)

        txt += ', no CRC treatments before FIT'
        flow = _add_counts(txt, flow, cdata)

    # Exclude patients with no record of CRC, but CRC-like treatments after FIT
    if ex_treatment_after_fit:

        e = ['colonic stent', 'local excision', 'radical resection', 'polypectomy', 'radiotherapy', 'chemotherapy']
        dfsub = cdata.events.loc[cdata.events.event.isin(e) & ~cdata.events.patient_id.isin(cdata.diagmin.patient_id)]
        mask = dfsub.start_date > dfsub.fit_date
        subj_ex = dfsub.loc[mask].patient_id.unique()
        print('{} individuals without CRC have some CRC-like treatments recorded after FIT date'.format(len(subj_ex)))
        subj = np.setdiff1d(cdata.fit.patient_id, subj_ex)

        cdata = _include_subj(cdata, subj)
        adata = _include_subj(adata, subj)
        
        txt += ', no CRC treatments after FIT in non-CRC group'
        flow = _add_counts(txt, flow, cdata)
    
    # Include data after introduction of buffer device?
    if buffer == 'fit-comment':
        # Get approx date from which stool pot comments were being added
        # so that having no comment implies it was buffer device
        date_from = cdata.fit.loc[cdata.fit.stool_pot == 1].fit_date.min()
        cdata.fit = cdata.fit.loc[(cdata.fit.fit_date >= date_from) & (cdata.fit.stool_pot == 0)]
        print('Number of FIT tests known to be in buffer device: ', cdata.fit.shape[0])
        print('Stool pot comment available since: ', date_from)

        subj_buf = cdata.fit.patient_id
        cdata = _include_subj(cdata, subj_buf)
        adata = _include_subj(adata, subj_buf)

    elif buffer == 'post-adoption':
        subj_buf = cdata.fit.loc[cdata.fit.fit_date >= "2021-10-01", 'patient_id']
        cdata = _include_subj(cdata, subj_buf)
        adata = _include_subj(adata, subj_buf)
        txt += ', buffer device'
        flow = _add_counts(txt, flow, cdata)
    
    elif buffer == 'pre-adoption':
        mask = cdata.fit.fit_date < '2021-07-01'
        cdata.fit = cdata.fit.loc[mask]
        subj_buf = cdata.fit.loc[mask, 'patient_id']
        cdata = _include_subj(cdata, subj_buf)
        adata = _include_subj(adata, subj_buf)
        txt += ', buffer device'
        flow = _add_counts(txt, flow, cdata)
    
    # Filter predictors
    #  Retain observations close to FIT date
    #  Ensure observations occur before CRC
    #  Drop observations available for less than npat patient
    if ppat_blood:
        npat = cdata.fit.patient_id.nunique()
        npat_blood = np.round(ppat_blood * npat).astype(int)
        print('\nIncluding bloods available for at least {}% of patients, equivalent to {} patients'.format(ppat_blood * 100, npat_blood))

    adata.bloods = filter_bloods(adata.bloods, cdata.fit, cdata.diagmin, days_before_fit_blood, days_after_fit_blood, npat_blood)
    adata.diag_codes = filter_codes(adata.diag_codes, cdata.fit, cdata.diagmin, days_before_fit_event, days_after_fit_event, npat_code, rmlast=True)
    adata.proc_codes = filter_codes(adata.proc_codes, cdata.fit, cdata.diagmin, days_before_fit_event, days_after_fit_event, npat_code, rmlast=True)
    adata.pres_codes = filter_codes(adata.pres_codes, cdata.fit, cdata.diagmin, days_before_fit_event, days_after_fit_event, npat_code)
    if adata.bmi.shape[0] > 0:
        adata.bmi = filter_bmi(adata.bmi, cdata.fit, cdata.diagmin, days_before_fit_event, days_after_fit_event)

    # Ensure each patient has at least days_before_fit_event days of potential clinical history
    if clinical_history:
        date_min = pd.to_datetime({'year':[2017], 'month':[1], 'day':[1]}).iloc[0]
        cdata.fit['days_from_start'] = (cdata.fit.fit_date - date_min).dt.days
        fit = cdata.fit
        subj_hist = fit.loc[fit.days_from_start >= days_before_fit_event, 'patient_id']
        cdata = _include_subj(cdata, subj_hist)
        adata = _include_subj(adata, subj_hist)
        txt += ', ' + str(days_before_fit_event) + ' days of clinical history'
        flow = _add_counts(txt, flow, cdata)

    # Exclude patients without certain bloods
    if coreblood:
        test_codes = ['HGB', 'PLT', 'MCV', 'WBC']
        for c in test_codes:
            df = adata.bloods.loc[adata.bloods.test_code == c]
            subj = df.patient_id.unique()
            cdata = _include_subj(cdata, subj)
            adata = _include_subj(adata, subj)
        txt += ', bloods'
        flow = _add_counts(txt, flow, cdata)
    elif coreblood_colofit:
        test_codes = ['PLT', 'MCV']
        for c in test_codes:
            df = adata.bloods.loc[adata.bloods.test_code == c]
            subj = df.patient_id.unique()
            cdata = _include_subj(cdata, subj)
            adata = _include_subj(adata, subj)
        txt += ', PLT and MCV'
        flow = _add_counts(txt, flow, cdata)       
    
    if save_data:
        save_coredata(cdata, run_path)
        save_additional_data(adata, run_path)
        flow.to_csv(run_path / INCLUSION_FILE, index=False)

    return cdata, adata, flow


def _include_subj(data, patient_id: pd.Series):
    """For each dataframe in data (a python dataclass),
    includes rows with patient_id identifiers given in patient_id
    """
    if data:
        for field in fields(data):
            df = getattr(data, field.name)
            if df.shape[0] > 0:
                df = df.loc[df.patient_id.isin(patient_id)]
                setattr(data, field.name, df)
    return data


def _add_counts(txt, flow, cdata):
    r = pd.DataFrame([[txt, cdata.fit.patient_id.nunique(), cdata.diagmin.patient_id.nunique()]], 
                      columns=['step', 'n_patient', 'n_patient_crc'])
    flow = pd.concat(objs=[flow, r], axis=0).reset_index(drop=True)
    return flow
