"""Get FIT values and corresponding clinical details"""
from dataprep.files import InputFiles, OutputFiles, DQFiles
from constants import DATA_DIR, DATACUT, PARQUET
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataprep import dq

f_in = InputFiles()
f_out = OutputFiles()
f_dq = DQFiles()

# Input files
FIT_FILE = f_in.fit
BLOODS_FILE = f_in.bloods

# Output files
FIT_OUT = f_out.fit
PLOT_FIT_HIST = f_dq.plot_fit_hist


def prepare_fit(run_path: Path, data_path: Path = DATA_DIR,
                save_data: bool = True, testmode: bool = False, gp_only: bool = False,
                apply_datacut: bool = True, correct_fit_sample_date: bool = True,
                correct_ida: bool = True):
    """Prepare FIT test values"""
    print('\n==== PREPARING FIT VALUES ====\n')
    run_path.mkdir(parents=True, exist_ok=True)

    # Read FIT data
    fit = _fit_get(data_path=data_path, apply_datacut=apply_datacut, testmode=testmode)

    # Correct FIT date (FIT sample date)
    # Note that in about 30% of cases, sample date is equal to date received (and can infer it was unknown)
    # In about another 30% of cases, it is equal to request date (and can also infer it was unknown)
    # Make sure that the 'fit_date' variable represents request date when it is known; otherwise set it to received date
    # AT comment 2025-07-31: 
    #  This code first computes the FIT request date (fit_request_date_corrected)
    #    When the FIT sample is received, the "fit_request_or_sample_date" is set to be the sample date,
    #    and the request date is copied into a separate "fit_request_date" column.
    #    To get all request dates, we can thus start with fit_request_or_sample_date, 
    #    and replace values those rows with rows from the "fit_request_date" column if the fit_request_date is available.
    #  Finally, after the FIT request date is computed, values in the "fit_date" column that are equal to request date
    #    are replaced with date received. This ensures that "fit_date" is either the sample date or date received,
    #    but not the date of the request.
    if correct_fit_sample_date:
        print('\nCorrecting FIT sample date...')
        gpreqs = pd.read_parquet(data_path / 'gpreqs')
        fit = fit.merge(gpreqs[['patient_id', 'icen', 'fit_request_date', 'fit_request_or_sample_date']], how='left')
        fit['fit_request_date_corrected'] = fit.fit_request_or_sample_date.copy()
        mask0 = ~fit.fit_request_date.isna()
        fit.loc[mask0, 'fit_request_date_corrected'] = fit.loc[mask0, 'fit_request_date']
        mask = fit.fit_request_date_corrected == fit.fit_date
        print("In {} observations, FIT sample collection date was equal to request date, and was replaced with sample received date".format(mask.sum()))
        fit.loc[mask, 'fit_date'] = fit.loc[mask, 'fit_date_received']
        test = fit.fit_date == fit.fit_date_received
        print("{} observations ({}%) have sample date unknown and was set equal to date received".format(test.sum(), test.mean()*100))

    if correct_ida:
        print('\nCorrecting symptom counts for IDA, low iron and anaemia...')
        # Ensure low_iron, ida and anaemia are mutually exclusive 
        mask = (fit.symptom_low_iron == 1) & (fit.symptom_anaemia == 1)
        fit.loc[mask, 'symptom_ida'] = 1
        fit.loc[mask, 'symptom_low_iron'] = 0
        fit.loc[mask, 'symptom_anaemia'] = 0

        mask = (fit.symptom_low_iron == 1) & (fit.symptom_ida == 1)
        fit.loc[mask, 'symptom_low_iron'] = 0

        mask = (fit.symptom_anaemia == 1) & (fit.symptom_ida == 1)
        fit.loc[mask, 'symptom_anaemia'] = 0

        test = fit[['symptom_low_iron', 'symptom_anaemia', 'symptom_ida']].value_counts(sort=False).reset_index()
        print('Check anaemia symptom counts\n\n', test)

    # Count total number of FIT tests before any filtering
    nfit = fit.patient_id.nunique()

    # Retain only GP FITs at this stage?
    # Some patients have a GP FIT not as their first FIT, in which case they'd be excluded later
    # However, if GP FITs are retained here, they won't be excluded
    if gp_only:
        print('Retaining only GP FITs...')
        fit = fit.loc[fit.gp == 1]
        print('Number of GP FITs: {}'.format(fit.shape[0]))
    
    # Clean FIT
    fit = _fit_clean(fit)

    # Get earliest FIT
    fit = _fit_earliest(fit)

    # Histogram of FIT test results
    _fit_plot(fit, run_path)

    if save_data:
        fit.to_csv(run_path / FIT_OUT, index=False)

    return fit, nfit


def _fit_get(data_path, apply_datacut=True, testmode=True):
    
    nrows = 5000 if testmode else None
    fit = pd.DataFrame()

    # FIT values from FIT table
    if PARQUET:
        df = pd.read_parquet(data_path / FIT_FILE)
        ind_cols = df.columns[df.columns.str.contains('^(?:fit10|gp|request_number|stool|already|specimen|inapp|symp)', regex=True)]
        df[ind_cols] = df[ind_cols].astype(int)
        print('\nData types in FIT data:\n\n', df.dtypes)
    else:
        df = pd.read_csv(data_path / FIT_FILE, nrows=nrows)
    #df = df.rename(columns={'dts': 'fit_date'})
    print('\nNumber of unclean FIT observations from LIMS: {}'.format(df.shape[0]))

    # Dates to datetime
    if not PARQUET:
        datecols = ['fit_date', 'fit_date_received', 'fit_date_authorised']
        dq.check_date_format(df, datecols=datecols, formats=['%Y-%m-%d %H:%M:%S.%f'])
        for c in datecols:
            df[c] = pd.to_datetime(df[c], format='%Y-%m-%d %H:%M:%S.%f')
            df[c] = df[c]   #.dt.normalize()  # Drop time
    df['src'] = 'lims'
    print('Min and max dates in LIMS FIT table: {}, {}'.format(df.fit_date.dt.date.min(), df.fit_date.dt.date.max()))

    # Apply datacut to fit_date?
    if apply_datacut:
        datacut = DATACUT.iloc[0]
        df = df.loc[df.fit_date <= datacut]
        print('Min and max dates in LIMS FIT table after datacut applied: {}, {}'.format(df.fit_date.dt.date.min(), 
                                                                                         df.fit_date.dt.date.max()))
    # Store
    fit = pd.concat(objs=[fit, df], axis=0)

    # Additional FIT values from bloods table?
    fit_from_bloods = False
    if fit_from_bloods:
        raise NotImplementedError
        df = pd.read_csv(data_path / BLOODS_FILE, nrows=nrows,
                         usecols=['patient_id', 'sample_collected_date_time', 'test_result', 'test_code'])
        df = df.loc[df.test_code.str.lower() == 'fitv']
        df = df.rename(columns={'sample_collected_date_time': 'fit_date', 'test_result': 'fit_val'})
        df = df[['patient_id', 'fit_date', 'fit_val']]
        print('\nNumber of unclean FIT observations from data warehouse: {}'.format(df.shape[0]))

        dq.check_date_format(df, datecols=['fit_date'], formats=['%d/%m/%Y %H:%M:%S'])
        df.fit_date = pd.to_datetime(df.fit_date, format='%d/%m/%Y %H:%M:%S')
        df.fit_date = df.fit_date   #.dt.normalize()  # Drop time
        df['src'] = 'datawarehouse'
        print('Min and max FIT dates in bloods table: {}, {}'.format(df.fit_date.dt.date.min(), df.fit_date.dt.date.max()))

        fit = pd.concat(objs=[fit, df], axis=0)

        # rm records in bloods table that overlap with records in LIMS
        fit0 = fit.loc[fit.src == 'lims']
        fit1 = fit.loc[fit.src == 'datawarehouse']
        fit1 = fit1.merge(fit0[['patient_id', 'fit_val', 'fit_date']], how='left', indicator=True)
        fit1 = fit1.loc[fit1._merge == 'left_only'].drop(labels=['_merge'], axis=1)
        print('\n{} FIT values were additionally found from data warehouse'.format(fit1.shape[0]))

        fit = pd.concat(objs=[fit0, fit1], axis=0)
    
    return fit


def _fit_clean(fit):

    if PARQUET:
        fit.fit_val_clean = fit.fit_val_clean.astype(float)
    
    # Clean 
    print('\nNumber of unclean FIT observations: {}'.format(fit.shape[0]))
    print('\nUnique values before cleaning:')
    for v in fit.fit_val.sort_values().unique():
        print(v)

    mask = fit.fit_val.fillna('').str.contains(r'\d', regex=True)
    fit = fit.loc[mask].copy()
    fit.fit_val = fit.fit_val.str.replace(r'[><\[\]\(\)]', '', regex=True)
    fit.fit_val = fit.fit_val.str.replace(r'\.$', '', regex=True)

    fit.fit_val = fit.fit_val.astype(float)
    fitsub = fit.loc[~fit.fit_val_clean.isna()]
    assert all(fitsub.fit_val == fitsub.fit_val_clean)

    # Apply detection limits
    fit.loc[fit.fit_val < 1.3, 'fit_val'] = 0
    fit.loc[fit.fit_val > 400, 'fit_val'] = 400

    # Round to integer?
    #fit.fit_val = fit.fit_val.round()
    print('\nNumber of clean FIT observations: {}'.format(fit.shape[0]))
    print('\nUnique values after cleaning')
    for v in fit.fit_val.sort_values().unique():
        print(v)
    
    return fit


def _fit_earliest(fit):
    """Get earliest FIT value for each 'patient_id'"""

    # Ensure FIT value has correct datatype
    fit.fit_val = fit.fit_val.astype(float)
    print('\nData types: \n{}'.format(fit.dtypes))

    # Check there are no missing values
    test = fit.isna().sum()
    print('\nNumber of missing values: \n{}'.format(test))
    #if any(test > 0):
    #    raise RuntimeError('FIT data has some missing values: double check')

    # Drop duplicate values for the same 'patient_id' at the same time point
    fit = fit.drop_duplicates().copy()
    print('\nNumber of FIT values after dropping duplicates: {}'.format(fit.shape[0]))

    # Explore number of FIT values per patient
    count = fit.groupby('patient_id').size().rename('nfit')
    print('\nNumber of patients with 1, 2, ... FIT observations:\n{}'.format(count.value_counts()))

    # Explore number of patients with multiple (different) FIT values at same time stamp
    fit['same_date_indicator'] = fit.duplicated(subset=['patient_id', 'fit_date'], keep=False)
    fit_dup = fit.loc[fit.same_date_indicator].copy()
    print('\nNumber of FIT observations that share dates: {}'.format(fit_dup.shape[0]))
    print('Number of patients with multiple FIT values on the same date: {}'.format(fit_dup['patient_id'].nunique()))

    # If multiple observations occur at the same date, replace with max value
    # However, if both GP and non-GP FITs are present, select maximum among GP FITs
    fit_dup = fit_dup.sort_values(by=['patient_id', 'fit_date', 'gp', 'fit_val'], 
                                  ascending=[False, False, False, False])
    fit_dup = fit_dup.groupby(['patient_id', 'fit_date']).first().reset_index()
    #fit_dup = fit_dup.groupby(['patient_id', 'fit_date'], as_index=False)['fit_val'].max()
    fit_nodup = fit.loc[~fit.same_date_indicator].copy()
    fit = pd.concat(objs=[fit_dup, fit_nodup], axis=0).sort_values(['patient_id', 'fit_date'])
    print('Number of FIT observations after replacing values on same date with max: {}'.format(fit.shape[0]))

    # Get earliest FIT value
    fit = fit.reset_index(drop=True)
    idx = fit.groupby('patient_id')['fit_date'].idxmin()
    fit = fit.loc[idx]
    print('Number of FIT observations after selecting earliest FIT value per patient: {}'.format(fit.shape[0]))
    print('Number of patients in FIT data after selecting earliest FIT value per patient: {}'.format(
        fit['patient_id'].nunique()))

    return fit


def _fit_plot(fit, run_path):
    """Plot basic histogram of FIT values"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(fit.fit_val, bins=100)
    ax[0].set(title='Histogram of FIT values')
    ax[1].hist(fit.fit_val[fit.fit_val > 1], bins=50)
    ax[1].set(title='Histogram of FIT values > 1 ug/g')
    plt.savefig(run_path / PLOT_FIT_HIST, dpi=150, bbox_inches='tight')
    plt.close()
