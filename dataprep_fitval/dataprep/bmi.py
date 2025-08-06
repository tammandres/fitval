from dataprep.files import InputFiles, OutputFiles, DQFiles
from constants import DATA_DIR, PARQUET
import pandas as pd
from pathlib import Path
from dataprep import dq
import numpy as np

f_in = InputFiles()
f_out = OutputFiles()
f_dq = DQFiles()

# Input files
BMI_FILE = f_in.bmi

# Output files
BMI_OUT = f_out.bmi
#BMI_DQ = f_dq.bmi_dq


def bmi(run_path: Path, fit: pd.DataFrame, data_path: Path = DATA_DIR, save_data: bool = True, testmode: bool = False):
    """Get maximum BMI within range of 'days_before_fit_event' and 'days_after_fit_event' of earliest FIT

    Some filtering is also done:
    * Unusually small or large values are excluded.
    * Data for patients for whom BMI changed more than 2-fold within that period are excluded
    """
    print('\n==== EXTRACTING BODY MASS INDEX ====')

    # Data
    nrows = 5000 if testmode else None
    if PARQUET:
        df = pd.read_parquet(data_path / BMI_FILE)
    else:
        df = pd.read_csv(data_path / BMI_FILE, nrows=nrows)

    # Date 
    if not PARQUET:
        dq.check_date_format(df, datecols=['observation_date_time'], formats=['%d/%m/%Y %H:%M:%S'])
        df.observation_date_time = pd.to_datetime(df.observation_date_time, format='%d/%m/%Y %H:%M:%S')

    # Retain patients with fit 
    df = df.loc[df.patient_id.isin(fit.patient_id)]
    print('\nTable has {} rows for {} patients'.format(df.shape[0], df.patient_id.nunique()))
    print(df.columns)

    # Explore observation codes
    print(df.observation_name.unique())

    # Get weight, length, bmi
    w = df.loc[df.observation_name.str.lower().str.contains('weight')].rename(columns={'observation_result':'observation_value'}).copy()
    l = df.loc[df.observation_name.str.lower().str.contains('height|length', regex=True)].rename(columns={'observation_result':'observation_value'}).copy()
    b = df.loc[df.observation_name.str.lower().str.contains('body mass index')].rename(columns={'observation_result':'observation_value'}).copy()

    print(w.observation_name.unique())
    print(l.observation_name.unique())
    print(b.observation_name.unique())

    # Simplify
    w.observation_name = 'weight'
    l.observation_name = 'height'
    b.observation_name = 'body mass index'

    # Explore values
    w.observation_value = w.observation_value.astype(float)
    l.observation_value = l.observation_value.astype(float)
    b.observation_value = b.observation_value.astype(float)

    perc = [0.001, 0.01,0.05,0.25,0.5,0.75,0.95,0.99, 0.999]
    print('\nBMI: \n{}'.format(b.observation_value.describe(percentiles=perc)))
    print('\nweight: \n{}'.format(w.observation_value.describe(percentiles=perc)))
    print('\nlength: \n{}'.format(l.observation_value.describe(percentiles=perc)))

    # Remove values that are clearly incorrect, not trying to correct these atm 
    b = b.loc[(b.observation_value > 10) & (b.observation_value < 300)]
    l = l.loc[(l.observation_value > 100) & (l.observation_value < 250)]
    print('\nBMI initial correction: \n{}'.format(b.observation_value.describe(percentiles=perc)))
    print('\nlength initial correction: \n{}'.format(l.observation_value.describe(percentiles=perc)))

    # Combine 
    df = pd.concat(objs=[w, l, b], axis=0)

    if save_data:
        df.to_csv(run_path / BMI_OUT, index=False)

    return df


def filter_bmi(df, fit, diagmin, days_before_fit_event, days_after_fit_event):

    # Retain observations close to FIT date
    df = df.merge(fit[['patient_id', 'fit_date']], how='inner')
    df['days_fit_to_event'] = df.observation_date_time - df.fit_date
    df.days_fit_to_event = df.days_fit_to_event.dt.days
    mask = (df.days_fit_to_event < days_after_fit_event) & (df.days_fit_to_event > -days_before_fit_event)
    df = df.loc[mask]
    print('\nNumber of observations within {} days before and {} days after FIT: {}'.format(days_before_fit_event, days_after_fit_event, df.shape[0]))
    print('\nTable has {} rows for {} patients'.format(df.shape[0], df.patient_id.nunique()))

    # Ensure all observations occurred before CRC date if CRC was present
    df = df.merge(diagmin[['patient_id', 'diagnosis_date']].rename(columns={'diagnosis_date':'crc_date'}), how='left')
    mask = df.observation_date_time >= df.crc_date 
    print('\nNumber of observations occurring after or at CRC date: {}'.format(mask.sum()))
    df = df.loc[~mask]
    print('\nNumber of observations left after dropping results after or at CRC date: {}'.format(df.shape[0]))

    return df


def max_bmi(df):

    # ==== Get length (min, max), and bmi (min, max) along with dates to check for data quality ====

    # Height
    h = df.loc[df.observation_name == 'height']
    h = h.sort_values(by=['patient_id', 'observation_value', 'observation_date_time'], ascending=[True, False, False])
    hmax = h.groupby('patient_id').first()[['observation_value', 'observation_date_time']].rename(columns={'observation_value':'length_max', 'observation_date_time':'length_max_date'}).reset_index()
    hmin = h.groupby('patient_id').last()[['observation_value', 'observation_date_time']].rename(columns={'observation_value':'length_min', 'observation_date_time':'length_min_date'}).reset_index()
    hsum = hmax.merge(hmin, how='outer', on='patient_id')

    # Bmi
    b = df.loc[df.observation_name == 'body mass index']
    b = b.sort_values(by=['patient_id', 'observation_value', 'observation_date_time'], ascending=[True, False, False])
    bmax = b.groupby('patient_id').first()[['observation_value', 'observation_date_time']].rename(columns={'observation_value':'bmi_max', 'observation_date_time':'bmi_max_date'}).reset_index()
    bmin = b.groupby('patient_id').last()[['observation_value', 'observation_date_time']].rename(columns={'observation_value':'bmi_min', 'observation_date_time':'bmi_min_date'}).reset_index()
    bsum = bmax.merge(bmin, how='outer', on='patient_id')

    # Combine
    bsum = bsum.merge(hsum, how='left', on='patient_id')
    print(bmin.shape, bmax.shape, bsum.shape)

    # Explore
    bsum['delta_bmi'] = np.abs(bsum.bmi_max - bsum.bmi_min)
    bsum['ratio_bmi'] = np.abs(bsum.bmi_max/bsum.bmi_min)
    bsum['delta_length'] = np.abs(bsum.length_max - bsum.length_min)
    bsum['delta_bmi_date'] = np.abs(bsum.bmi_max_date - bsum.bmi_min_date)
    bsum['delta_bmi_per_day'] = (bsum.delta_bmi/(bsum.delta_bmi_date.dt.days+1))
    bsum = bsum.sort_values('ratio_bmi', ascending=False)

    # If min and max recorded BMI ratio greater than 2 -- do not use
    s = bsum.loc[bsum.ratio_bmi > 2]
    print('\nRecords with min and max BMI ratio greater than 2:')
    print(s)
    
    # Extract max BMI
    df = bsum.loc[bsum.ratio_bmi <= 2]
    df = df[['patient_id', 'bmi_max', 'bmi_max_date']].rename(columns={'bmi_max_date':'event_date'})
    print('\nSnapshot of BMI data: \n{}'.format(df.head()))
    print('\nSummary of BMI: \n{}'.format(df.bmi_max.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])))

    # Save
    #if save_data:
    #    df.to_csv(run_path / BMI_OUT, index=False)
    #    bsum.to_csv(run_path / BMI_DQ, index=False)
    
    return df, bsum
    