"""Get TNM staging
* T-value and associate N and M values for all individuals with CRC, from pathology reports
* T/N/M values (if any given) with dates, from imaging and pathology reports
"""
import pandas as pd
from pathlib import Path
from constants import DATA_DIR, PARQUET
from dataprep import dq
from dataprep.files import InputFiles, OutputFiles, DQFiles

f_in = InputFiles()
f_out = OutputFiles()
dq_out = DQFiles()

# Input files
HIST_FILE = f_in.hist
IMG_FILE = f_in.imaging
CRC_FROM_PATH_FILE = f_in.crc_from_path

# Outout files
DIAGMIN_OUT = f_out.diagmin
MATCHES_TNM = dq_out.matches_tnm
MATCHES_TNM_EX = dq_out.matches_tnm_ex
MATCHES_CRC = dq_out.matches_crc
REPORTS_OUT = dq_out.reports


def stage(run_path: Path, fit: pd.DataFrame, diagmin: pd.DataFrame, save_data: bool = True, 
          testmode: bool = False, days_stage_after_diag: int = 180):
    """Get TNM staging"""

    print('\n==== EXTRACTING TNM STAGE ====\n')

    # Extract TNM stage from imaging and pathology reports that mention CRC
    df = _get_stage(fit, diagmin)

    # Get max T-stage, and associated N and M values, and add it to CRC date table
    diagmin = _max_t_stage(df, diagmin, days_stage_after_diag=days_stage_after_diag)

    # Get event logs for TNM stage (event: if any of the T, N or M values is recorded)
    events = df[['patient_id', 'T', 'N', 'M', 'report_date', 'report_type']].copy()
    events = events.dropna(how='all', subset=['T', 'N', 'M'])
    events['event'] = 'TNM staging'
    events = events.rename(columns={'report_date': 'start_date'})
    events = events[['patient_id', 'event', 'start_date', 'report_type', 'T', 'N', 'M']]

    if save_data:
        df.to_csv(run_path / REPORTS_OUT, index=False)
        diagmin.to_csv(run_path / DIAGMIN_OUT, index=False)
        #matches_crc.to_csv(run_path / MATCHES_CRC, index=False)
        #matches_tnm.to_csv(run_path / MATCHES_TNM, index=False)
        #check_rm.to_csv(run_path / MATCHES_TNM_EX, index=False)

    return diagmin, events
    

def _get_stage(fit, diagmin, data_dir=DATA_DIR, testmode=False, rm_historical=False):
    """Extract T, N and M values for imaging and pathology reports
    
    Args:
        fit: dataframe with FIT values
        diagmin: dataframe with earliest CRC dates
        data_dir: where csv files are located
        testmode: if True, code is run on a small number of reports (for testing)
        rm_historical: remove T/N/M values that may be historical
    """

    # ==== Get pathology and imaging reports ====
    df = pd.DataFrame()

    # Pathology
    if PARQUET:
        df1 = pd.read_parquet(data_dir / HIST_FILE)
        crc = pd.read_parquet(data_dir / CRC_FROM_PATH_FILE)
        crc.crc_nlp = crc.crc_nlp.astype(int)
    else:
        df1 = pd.read_csv(data_dir / HIST_FILE)
        crc = pd.read_csv(data_dir / CRC_FROM_PATH_FILE)
    
    print(df1.shape)
    df1 = df1.drop_duplicates()
    print(df1.shape)
    crc = crc[['patient_id', 'received_date', 'authorised_date', 'sent_date', 'snomed_t', 'snomed_m', 'crc_nlp']].drop_duplicates()
    df1 = df1.merge(crc, how='outer', on=['patient_id', 'received_date', 'authorised_date', 'sent_date', 'snomed_t', 'snomed_m'])
    print(df1.shape)

    df1 = df1[['patient_id', 'received_date', 'snomed_t', 'T_pre', 'T', 'N', 'M', 'crc_nlp']].rename(columns={'received_date': 'report_date'})
    df1 = df1.loc[df1.patient_id.isin(fit.patient_id)]
    df1['report_type'] = 'pathology'
    df1['report_date'] = pd.to_datetime(df1['report_date'], format='%Y-%m-%d')

    df1 = df1.loc[df1.crc_nlp==1]  ## Retain crc reports only

    df = pd.concat(objs=[df, df1], axis=0)

    # Imaging
    if PARQUET:
        df2 = pd.read_parquet(data_dir / IMG_FILE)
    else:
        df2 = pd.read_csv(data_dir / IMG_FILE)
    df2 = df2[['patient_id', 'imaging_report_date', 't_pre', 't', 'n', 'm']].rename(columns={'imaging_report_date': 'report_date'})
    df2 = df2.rename(columns={'t': 'T', 'n': 'N', 'm': 'M', 't_pre': 'T_pre'})
    df2 = df2.loc[df2.patient_id.isin(fit.patient_id)]
    df2['report_type'] = 'imaging'
    df2['report_date'] = pd.to_datetime(df2['report_date'], format='%Y-%m-%d')

    ## Don't add imaging and endoscopy staging atm - as not known if CRC report
    #df = pd.concat(objs=[df, df2], axis=0)
    # don't have crc_nlp for img reports, but can assume that if occur close to diagnosis and mentions T-stage prob crc?

    # Add indicator for whether individual had CRC
    df['crc'] = 0
    df.loc[df.patient_id.isin(diagmin.patient_id), 'crc'] = 1
    df = df.loc[df.crc == 1]

    # Retain reports that mention current CRC?

    return df


def _max_t_stage(df, diagmin, days_stage_after_diag):
    """Extract maximum T value, and associated N and M values, for each patient
    
    Args
        df: dataframe that contains imaging and pathology reports,
            T/N/M values for each report, and patient identifiers
        diagmin: dataframe that contains earliest CRC dates for each patient
        days_stage_after_diag: maximum number of days in which T value must occur after earliest CRC date
            to be included.
    """

    # Get T-stages
    cols = ['patient_id', 'report_date', 'T_pre', 'T', 'N', 'M', 'crc', 'report_type']
    dft = df[cols].drop_duplicates().dropna(subset=['T'])

    # Explore pathology reports that mention CRC: do some mention CRC but not T-stage? (yes)
    explore = False
    if explore:
        dsub = diagmin.loc[diagmin.diagnosis_source == 'pathology']
        dsub = dsub.merge(dft, how='left', on=['patient_id'])
        subj = dsub.patient_id.unique()
        for i, s in enumerate(subj):
            rsub = dsub.loc[dsub.patient_id == s, 'safe_report'].str.replace('\r', '\n')
            for j, r in enumerate(rsub):
                print('\n====Reports for individual {}, id {}'.format(i, s))
                print('\n----Report {}\n'.format(j))
                print(r)
                print('\n====End of reports for this individual.')
    
    # ==== Retain T-stages close to CRC date ====

    # Date col
    datecol = 'report_date'

    # Add diagnosis date to reports
    print(dft.shape)
    dft = dft.merge(diagmin[['patient_id', 'diagnosis_date']], how='left')
    print(dft.shape)

    # Ensure lowercase
    cols = ['T_pre', 'T']
    for c in cols:
        dft[c] = dft[c].str.lower()

    # Retain T-stages given within 'days_stage_after_diag' after diagnosis date
    dft['days_diag_to_report'] = (dft[datecol] - dft.diagnosis_date).dt.days
    print('\n')
    print(dft[['days_diag_to_report']].describe())
    mask = (dft.days_diag_to_report <= days_stage_after_diag) & (dft.days_diag_to_report >= 0)
    dft = dft.loc[mask]
    print(dft.shape)

    # Summary
    ntot = diagmin.patient_id.nunique()
    nt = dft.patient_id.nunique()
    print('\n{} ({:.2f}%) of {} CRC patients have pathological T-stage recorded near diag date'.format(nt, nt / ntot * 100,
                                                                                                       ntot))

    # ==== Get max T stage ====
    stage = pd.DataFrame()

    # Convert x and is, so T stage can be sorted in descending order
    repl = {'x': '-1', 'is': '0.5'}
    dft['T_sort'] = dft['T'].replace(repl)
    dft = dft.sort_values(by=['patient_id', 'T_sort'], ascending=[True, False])

    # Get max pathological stage
    s = dft.loc[dft.report_type == 'pathology'].groupby('patient_id')[['T', 'N', 'M', datecol]].first().reset_index()
    s = s.rename(columns={datecol: 'stage_date'})
    s['stage_source'] = 'pathology'
    stage = pd.concat(objs=[stage, s], axis=0)
    stage_path = s

    # Get max img stage -- if path not available
    mask = (dft.report_type == 'imaging') & (~dft.patient_id.isin(stage_path.patient_id))
    s = dft.loc[mask].groupby('patient_id')[['T', 'N', 'M', datecol]].first().reset_index()
    s = s.rename(columns={datecol: 'stage_date'})
    s['stage_source'] = 'imaging'
    stage = pd.concat(objs=[stage, s], axis=0)

    # Explore
    s = stage.groupby('stage_source')['patient_id'].nunique()
    print('\nNumber of patient_ids with T stage by report source: \n{}'.format(s))

    # Add maximum T stage, and associated N and M values, to diagnosis table
    diagmin = diagmin.merge(stage, how='left')
    print(diagmin.head())
    nmis = diagmin['T'].isna().sum()
    nsub = diagmin.patient_id.nunique()
    print('Check: diag table has {} patient_ids and {} rows'.format(diagmin.patient_id.nunique(), diagmin.shape[0]))
    print('T stage not known for {} ({:.2f}%) of {} patients'.format(nmis, nmis / nsub * 100, nsub))

    return diagmin
    