"""Get all blood test results"""
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
BLOODS_FILE = f_in.bloods

# Output files
BLOODS_OUT = f_out.bloods
BLOODS_HILO_OUT = f_out.bloods_hilo
BLOODS_SUM = f_dq.bloods_sum


def bloods(run_path: Path, fit: pd.DataFrame, data_path: Path = DATA_DIR, 
           save_data: bool = True, testmode: bool = False, npat: int = 100, 
           days_before_fit_blood: int = 365, days_after_fit_blood: int = 30):
    """Get all blood test results within range from earliest FIT"""
    print('\n==== EXTRACTING AND CLEANING BLOOD TEST RESULTS ====\n')
    
    # Read blood test results, retain those within range of earliest FIT date
    df = _get_bloods(data_path=data_path, fit=fit, testmode=testmode)

    # Clean, e.g. remove values not numeric, replace >10 with 10 etc.
    df = _clean_bloods(df)

    # Explore test code - test name - test unit combinations
    s = _explore_bloods(df)

    # Retain bloods available for at least npat patients - to reduce file size
    mask = s.nsub < npat
    s_sub = s.loc[~mask]
    df = df.merge(s_sub, how='inner')
    print('Number of tests available for less than {} patients: {}'.format(npat, mask.sum()))
    print('\nTests available for less than {} patients: \n{}'.format(npat, s.loc[mask]))
    print('\nTests available for {} patients or more: \n{}'.format(npat, s.loc[~mask]))
    print('Number of observations left in bloods table after dropping rare tests: {}'.format(df.shape[0]))

    # Retain bloods within certain range of FIT, to reduce file size
    df = df.merge(fit[['patient_id', 'fit_date']], how='inner')
    print('Number of observations in bloods table for patients that had FIT: {}'.format(df.shape[0]))
    df['days_fit_to_blood'] = df.sample_collected_date_time - df.fit_date
    df.days_fit_to_blood = df.days_fit_to_blood.dt.days

    mask = (df.days_fit_to_blood < days_after_fit_blood) & (df.days_fit_to_blood > -days_before_fit_blood)
    df = df.loc[mask]
    print('Number of observations within {} days before and {} days after FIT: {}'.format(days_before_fit_blood, days_after_fit_blood, df.shape[0]))

    # Additionally get high-low values for specific bloods
    #df_hilo = bloods_high_low(df, demo)
    #df_hilo.to_csv(run_path / BLOODS_HILO_OUT, index=False)

    if save_data:
        df.to_csv(run_path / BLOODS_OUT, index=False)
        s.to_csv(run_path / BLOODS_SUM, index=False)
        
    return df, s


def _get_bloods(data_path: Path, fit: pd.DataFrame, testmode: bool = False):
    """Read blood test results, retain those that belong to individuals with FIT values"""

    # Read
    nrows = 50000 if testmode else None
    if PARQUET:
        df = pd.read_parquet(data_path / BLOODS_FILE)
    else:
        df = pd.read_csv(data_path / BLOODS_FILE, nrows=nrows)
    df = df.rename(columns={'test_collection_date_time': 'sample_collected_date_time'})

    # Dates to datetime (only if not parquet format, as o.w. the dates are already in correct format)
    if not PARQUET:
        dq.check_date_format(df, datecols=['sample_collected_date_time'], formats=[DATETIME_FORMAT])
        df.sample_collected_date_time = pd.to_datetime(df.sample_collected_date_time, format=DATETIME_FORMAT)

        dq.check_date_format(df, datecols=['result_date_time'], formats=[DATETIME_FORMAT])
        df.result_date_time = pd.to_datetime(df.result_date_time, format=DATETIME_FORMAT)
    print('Number of observations in bloods table: {}'.format(df.shape[0]))

    # Drop duplicates
    df = df.drop_duplicates()
    print('Number of observations in bloods table after dropping duplicates: {}'.format(df.shape[0]))

    # Retain patient_ids with FIT
    df = df.loc[df.patient_id.isin(fit.patient_id)]

    # Drop FIT from bloods
    mask = ~df.test_code.str.lower().isin(['fitv', 'fitr'])
    df = df.loc[mask]
    print('Number of observations in bloods after dropping FIT: {}'.format(df.shape[0]))

    # Remove observations with empty strings (if any)
    mask = df.test_result == ''
    df = df.loc[~mask]
    print('Number of observations with empty strings: {}'.format(mask.sum()))

    # Remove of observations with missing test result or test date (if any)
    mask = df.test_result.isna() | df.sample_collected_date_time.isna()
    print('Number of observations with missing test result: {}'.format(mask.sum()))
    df = df.loc[~mask]

    return df


def filter_bloods(df, fit, diagmin, days_before_fit_blood, days_after_fit_blood, npat):
    """Ensure that tests ocurred within [days_before_fit_blood, days_after_fit_blood] range
    from the earliest FIT date, and before CRC

    NB. days_after_fit_blood, days_before_fit_blood: positive integers
    """

    # Time from FIT to blood test in days
    if 'fit_date' in df.columns:
        df = df.drop(labels=['fit_date'], axis=1)
    df = df.merge(fit[['patient_id', 'fit_date']], how='inner')
    print('Number of observations in bloods table for patients that had FIT: {}'.format(df.shape[0]))
    df['days_fit_to_blood'] = df.sample_collected_date_time - df.fit_date
    df.days_fit_to_blood = df.days_fit_to_blood.dt.days

    # Retain observations close to FIT date
    mask = (df.days_fit_to_blood < days_after_fit_blood) & (df.days_fit_to_blood > -days_before_fit_blood)
    df = df.loc[mask]
    print('Number of observations within {} days before and {} days after FIT: {}'.format(days_before_fit_blood, days_after_fit_blood, df.shape[0]))

    # Ensure all blood tests occurred before CRC date if CRC was present
    df = df.merge(diagmin[['patient_id', 'diagnosis_date']].rename(columns={'diagnosis_date':'crc_date'}), how='left')
    mask = df.sample_collected_date_time >= df.crc_date 
    print('Number of observations in bloods table occurring after or at CRC date: {}'.format(mask.sum()))
    df = df.loc[~mask]
    print('Number of observations left in bloods table after dropping results after or at CRC date: {}'.format(df.shape[0]))

    # Drop tests available for less than npat patients
    s = _explore_bloods(df)
    if npat is not None:
        if 'nsub' in df.columns:
            df = df.drop(labels=['nsub'], axis=1)
        mask = s.nsub < npat
        s_sub = s.loc[~mask]
        df = df.merge(s_sub, how='inner', on=['test_code', 'test_name', 'test_units'])
        print('Number of tests available for less than {} patients: {}'.format(npat, mask.sum()))
        print('\nTests available for less than {} patients: \n{}'.format(npat, s.loc[mask]))
        print('\nTests available for {} patients or more: \n{}'.format(npat, s.loc[~mask]))
        print('Number of observations left in bloods table after dropping rare tests: {}'.format(df.shape[0]))

    return df


def _clean_bloods(df):

    # ==== REMOVE OR CLEAN OBSERVATIONS ====

    # Remove observations that contain the following patterns
    #  Letters, brackets, / or \, +, comma, only nonword characters
    print('\n---Removing observations with certain patterns')
    patterns = [r'[a-zA-Z]', r'\[\]\)\(', r'\\|\/', r'\+', r'^\W*$', r'\d+\s*\-\s*\d+', r'\?', r'\*']
    for pat in patterns:
        print('\nPattern: {}'.format(repr(pat)))
        mask = df.test_result.str.contains(pat, regex=True)
        print('Number of observations with pattern: {}'.format(mask.sum()))
        print('Unique values with pattern: \n{}'.format(df.loc[mask].test_result.unique()))
        df = df.loc[~mask]
        print('Number of observations left in bloods table: {}'.format(df.shape[0]))

    # Clean observations that contain the following patterns
    #  < or ? or *, ?, or = or empty space
    print('\n---Cleaning observations with certain patterns')
    patterns = [r'<|>', r'\=', r'\s', '%', r'\,', r'\.$']
    for pat in patterns:
        print('\nPattern: {}'.format(repr(pat)))
        mask = df.test_result.str.contains(pat, regex=True)
        print('Number of observations with pattern: {}'.format(mask.sum()))
        print('Unique values with pattern: \n{}'.format(df.loc[mask].test_result.unique()))
        df.test_result = df.test_result.str.replace(pat, '', regex=True)
        df.test_result = df.test_result.replace({'':np.nan})
        df = df.dropna(subset=['test_result'])  # If observation consisted of the string, drop it entirely
        print('Number of observations left in bloods table: {}'.format(df.shape[0]))

    # Any observations left that do not look like numbers?
    mask = ~df.test_result.str.contains(r'\-?\d?\.\d{1,}|^-?\d{1,}$', regex=True)
    print('\nNumber of observations left that do not look like numbers: {}'.format(mask.sum()))
    print('Unique values that do not look like numbers: \n{}'.format(df.loc[mask].test_result.unique()))

    # Convert to float
    df.test_result = df.test_result.astype(float)

    # Check missing values
    test = df.isna().sum()
    print('\nNumber of missing values in overall table: \n{}'.format(test))

    # ==== Rename test codes and units that mean the same thing ====

    #  Get test code - test name - test unit combinations
    s = _explore_bloods(df)

    # Explore tests with duplicate codes
    tmp = s.loc[s.test_code.duplicated(keep=False)].copy()
    print('\nTests with duplicate test codes but different test names or units: \n{}'.format(tmp))

    # If a single test code is associated with multiple unit names, give a single name to units that mean the same thing
    print('\nUnique values of units for tests with duplicate codes: \n{}'.format(tmp.test_units.unique()))
    rep = {'UMOL/L':'umol/L', 
           'MICROGRAM/G':'microgram/g', 
           'mmol/l':'mmol/L',
           'mmol/24':'mmol/24Hr'
           }
    df.test_units = df.test_units.replace(rep)
    s = df.groupby(['test_code', 'test_name', 'test_units'])['patient_id'].nunique().rename('nsub').reset_index()
    tmp = s.loc[s.test_code.duplicated(keep=False)].copy()
    print('\nUnique values of units for tests with duplicate codes (after fixing): \n{}'.format(tmp.test_units.unique()))

    # Replace unusual codes with test names -- so that test_code and test_units would uniquely define a test
    tmp = s.loc[s.test_code.fillna('').str.contains('^\W*$')].copy()
    print('\nTests with unusual test codes: \n{}'.format(tmp))
    if not tmp.empty:
        mask = df.test_code.str.contains('^\W*$')
        df.loc[mask, 'test_code'] = 'uncoded_' + df.loc[mask, 'test_name']
        print('\nTests with unusual test codes now have codes: \n{}'.format(df.loc[mask, 'test_code'].unique()))

    # Explore tests with duplicate codes again
    s = df.groupby(['test_code', 'test_units'])['patient_id'].nunique().rename('nsub').reset_index()
    tmp = s.loc[s.test_code.duplicated(keep=False)].copy()
    print('\n(After fixes) Tests with duplicate test codes but different units: \n{}'.format(tmp))

    # Explore tests available after fixing
    s = df.groupby(['test_code', 'test_units'])['patient_id'].nunique().rename('nsub').reset_index()
    pd.set_option('display.min_rows', 1000, 'display.max_rows', 1000)
    print('Number of tests: {}'.format(s.shape[0]))
    print('\n(After fixes) Tests available: \n{}'.format(s))

    return df


def _explore_bloods(df):

    # Explore test codes, names and units
    s = df.groupby(['test_code', 'test_name', 'test_units'])['patient_id'].nunique().rename('nsub').reset_index()
    pd.set_option('display.min_rows', 1000, 'display.max_rows', 1000)
    print('Number of tests: {}'.format(s.shape[0]))
    #print('\nTests available: \n{}'.format(s))

    return s


def bloods_high_low(df: pd.DataFrame, demo: pd.DataFrame, run_path: Path, save_data: bool = True):
    """Get high-low indicators for certain bloods"""

    #df.loc[df.test_code.str.lower().str.contains('crp')][['test_code', 'test_name', 'test_units']].drop_duplicates()
    #df.loc[df.test_name.str.lower().str.contains('prot')][['test_code', 'test_name', 'test_units']].drop_duplicates()
    #df.loc[df.test_name.str#df.loc[df.test_code.str.lower().str.contains('zcrp')].test_result.describe(percentiles=[0.01,0.05,0.25,0.5,0.75]).lower().str.contains('prot')][['test_code', 'test_name', 'test_units']].drop_duplicates()


    # Rules for basic blood tests
    # Based on Withrow et al, https://doi.org/10.1186/s12916-022-02272-w
    ref = pd.DataFrame()

    r = pd.DataFrame([['HGB','g/L', 130, '<', 'M', 'low haemoglobin']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['HGB','g/L', 120, '<', 'F', 'low haemoglobin']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['PLT','x10*9/L', 400, '>', np.nan, 'high platelets']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['WBC','x10*9/L', 11, '>', np.nan, 'high white cells']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['MCH','pg', 27.4, '<', np.nan, 'low mean cell haemoglobin']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['MCV','fl', 80, '<', np.nan, 'low mean cell volume']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['CFER','microg/L', 20, '<', np.nan, 'low serum ferritin']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['CFER','microg/L', 350, '>=', np.nan, 'high serum ferritin']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['ZCRP','mg/L', 10, '>', np.nan, 'high C-reactive protein']])
    ref = pd.concat(objs=[ref, r], axis=0)

    r = pd.DataFrame([['CRPP','mg/L', 10, '>', np.nan, 'high C-reactive protein']])
    ref = pd.concat(objs=[ref, r], axis=0)

    ref = ref.reset_index(drop=True)
    ref.columns = ['test_code', 'test_units', 'ref', 'comparison', 'gender', 'meaning']

    # Double check that all test codes and units are present 
    u = df[['test_code', 'test_units']].drop_duplicates()
    for i,row in ref.iterrows():
        test0 = np.isin([row.test_code], u.test_code)[0]
        if test0:
            test1 = (row.test_units == u.loc[u.test_code == row.test_code, 'test_units']).iloc[0]
        else:
            test1 = False
        print('test_code {} present: {}, test_units {} present: {}'.format(row.test_code, test0, row.test_units, test1))

    
    # Identify high/low results for bloods of interest 
    repl = {'HGB':'haemoglobin', 'PLT':'platelets', 'WBC':'white cells', 'MCH':'mean cell haemoglobin',
            'MCV':'mean cell volumne', 'CFER':'serum ferritin', 'ZCRP':'C-reactive protein',
            'CRPP':'C-reactive protein'}
    ref = ref.loc[ref.test_code.isin(df.test_code.unique())]

    bsub = pd.DataFrame()

    ## HGB -- as ref range depends on gender
    b = df.merge(demo[['patient_id', 'gender']], on=['patient_id'], how='left')
    b = ref.loc[ref.test_code=='HGB'].merge(b, on=['test_code', 'test_units', 'gender'], how='left')
    print(b.shape)

    b['test_result_bin'] = 'normal ' + b.test_code.replace(repl)
    mask = (b.comparison=='<')&(b.test_result < b.ref)
    b.loc[mask, 'test_result_bin'] = b.loc[mask, 'meaning']

    mask = (b.comparison=='>')&(b.test_result > b.ref)
    b.loc[mask, 'test_result_bin'] = b.loc[mask, 'meaning']

    mask = (b.comparison=='>=')&(b.test_result >= b.ref)
    b.loc[mask, 'test_result_bin'] = b.loc[mask, 'meaning']

    bsub = pd.concat(objs=[bsub, b], axis=0)

    ## Other tests -- ref range does not depend on gender
    b = ref.loc[ref.test_code!='HGB'].merge(df, on=['test_code', 'test_units'], how='left')
    print(b.shape)

    b['test_result_bin'] = 'normal ' + b.test_code.replace(repl)
    mask = (b.comparison=='<')&(b.test_result < b.ref)
    b.loc[mask, 'test_result_bin'] = b.loc[mask, 'meaning']

    mask = (b.comparison=='>')&(b.test_result > b.ref)
    b.loc[mask, 'test_result_bin'] = b.loc[mask, 'meaning']

    mask = (b.comparison=='>=')&(b.test_result >= b.ref)
    b.loc[mask, 'test_result_bin'] = b.loc[mask, 'meaning']

    bsub = pd.concat(objs=[bsub, b], axis=0)

    bsub = bsub[['patient_id', 'test_code', 'test_result_bin']].reset_index(drop=True)  #dropna() -- no, to count missingness
    print(bsub.shape)
    print(bsub[['test_code', 'test_result_bin']].drop_duplicates())

    if save_data:
        bsub.to_csv(run_path / BLOODS_HILO_OUT, index=False)

    return bsub
