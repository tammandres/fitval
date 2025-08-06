"""Get imaging events: e.g. MRI scans and their dates"""
import pandas as pd
from dataprep import dq
from constants import DATA_DIR, DATAPREP_DIR, PARQUET
from dataprep.files import InputFiles, IMG_CODE_FILE
from pathlib import Path

f_in = InputFiles()
IMG_FILE = f_in.imaging


def imaging_events(fit: pd.DataFrame, data_dir: Path = DATA_DIR, code_dir: Path = DATAPREP_DIR):
    """Get imaging events and dates
    Relying on a list of imaging codes prepared by Dr Helen Jones for the NIHR HIC Colorectal Cancer theme
    """
    print('\n==== Extracting imaging events and dates ====\n')
    
    # Get imaging types to be included
    def list_to_string(scans):
        scans = ["'" + s + "'" for s in scans]
        scans = ",".join(scans)
        return scans

    df_img = pd.read_excel(code_dir / IMG_CODE_FILE)
    df_img = df_img.loc[df_img['needed for NIHR HIC CRC'] != 'no']

    scans_diag = df_img.loc[df_img['needed for NIHR HIC CRC'] == 'diagnosis', 'imaging_code'].to_numpy()
    scans_diag = list_to_string(scans_diag)

    scans_surv = df_img.loc[df_img['needed for NIHR HIC CRC'] == 'surveillance', 'imaging_code'].to_numpy()
    scans_surv = list_to_string(scans_surv)

    scans_meta = df_img.loc[df_img['needed for NIHR HIC CRC'] == 'metastasis', 'imaging_code'].to_numpy()
    scans_meta = list_to_string(scans_meta)

    scans_all = df_img.loc[df_img['needed for NIHR HIC CRC'] != 'maybe'].imaging_code.to_numpy()
    scans_all = list_to_string(scans_all)

    print('\nImaging codes for diagnosis: {}'.format(scans_diag))
    print('\nImaging codes for surveillance: {}'.format(scans_surv))
    print('\nImaging codes for metastasis: {}'.format(scans_meta))
    print('\nAll relevant imaging codes: {}'.format(scans_all))
    print(df_img.head())

    # Get imaging events
    if PARQUET:
        df = pd.read_parquet(data_dir / IMG_FILE)
    else:
        df = pd.read_csv(data_dir / IMG_FILE)
    print(df.head())

    #dq.check_date_format(df, datecols=['imaging_date'], formats=['%d/%m/%Y %H:%M:%S'])
    #df.imaging_date = pd.to_datetime(df.imaging_date, format='%d/%m/%Y %H:%M:%S')
    if not PARQUET:
        dq.check_date_format(df, datecols=['imaging_date'], formats=['%Y-%m-%d'])
    df.imaging_date = pd.to_datetime(df.imaging_date, format='%Y-%m-%d')
    print('Number of imaging events: {}'.format(df.shape[0]))

    df = df.loc[df.imaging_code.isin(df_img.imaging_code)]
    print('Number of imaging events with relevant imaging codes: {}'.format(df.shape[0]))

    df = df[['patient_id', 'imaging_date', 'imaging_code']].rename(columns={'imaging_date': 'start_date'})
    df['event'] = 'scan'

    # Retain patient_ids w FIT
    # img_all = df.copy()
    df = df.loc[df.patient_id.isin(fit.patient_id)]

    return df
