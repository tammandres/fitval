"""Get demographics: gender, age, IMDD, etc"""
from constants import DATA_DIR, PARQUET
from dataprep.files import InputFiles, OutputFiles
import pandas as pd
from pathlib import Path
from dataprep import dq

f_in = InputFiles()
f_out = OutputFiles()

DEMO_FILE = f_in.demo
DEMO_OUT = f_out.demo


def demographics(fit, run_path: Path, data_path: Path = DATA_DIR, save_data: bool = True):
    """Get demographics:
    gender, ethnicity, maximum index of multiple deprivation, age at earliest FIT test, death date,
    date last alive for patients not known to have died.

        fit: DataFrame of FIT values
        run_path: where to save results
        data_path: path to FIT dataset csv files
        save_data: if True, data is saved to disk

    Assumes specific column names, e.g. individual is named as 'patient_id' in the data.
    """
    print('\n==== PREPARING DEMOGRAPHICS ====\n')
    run_path.mkdir(parents=True, exist_ok=True)

    # Demographics
    if PARQUET:
        df = pd.read_parquet(data_path / DEMO_FILE)
    else:
        df = pd.read_csv(data_path / DEMO_FILE)
    print('Number of patients in demographics table: {}'.format(df['patient_id'].nunique()))
    print('Columns: {}'.format(df.columns))

    # Date to datetime
    df['birth_date'] = df.year_of_birth.astype(str) + '-' + df.month_of_birth.astype(str).str.zfill(2)
    dq.check_date_format(df, datecols=['birth_date'], formats=['%Y-%m'])
    df.birth_date = pd.to_datetime(df.birth_date, format='%Y-%m')

    if not PARQUET:
        dq.check_date_format(df, datecols=['death_date'], formats=['%Y-%m-%d'])
    df['death_date'] = pd.to_datetime(df.death_date, format='%Y-%m-%d')
    #dq.check_date_format(df, datecols=['birth_date', 'death_date'], formats=['%Y-%m', '%Y-%m-%d'])
    #df = dq.to_datetime_multi(df, datecols=['birth_date', 'death_date'], formats=['%Y-%m', '%Y-%m-%d'] )
    print(df.head())

    # Retain patients with FIT
    df = df.merge(fit[['patient_id', 'fit_date']], how='inner')
    print('\nNumber of patients in demographics table: {}'.format(df['patient_id'].nunique()))

    # Check missing values
    test = df.isna().sum(axis=0)
    print('\nNumber of missing values per column: \n{}'.format(test))

    # Get age at fit test
    df['age_at_fit'] = (df.fit_date - df.birth_date).dt.days/365

    # Get max imdd (as some individuals have multiple IMDD values)
    df = df.rename(columns={'index_of_multiple_deprivation': 'imdd'})
    s = df.groupby('patient_id').nunique().reset_index(drop=True)
    test = (s > 1).sum(axis=0)
    test = test[test > 0]
    print('\nNumber of individuals with multiple records for some variables: \n{}'.format(test))

    i = df.groupby('patient_id')['imdd'].max().rename('imdd_max').reset_index()
    df = df.merge(i, how='left', on='patient_id').drop(labels=['imdd'], axis=1).drop_duplicates()

    # Get last contact date for patients who are alive (not known to have died)
    #df['last_alive_date'] = df[['last_contact', 'spine_check_date']].max(axis=1)
    #df.loc[~df.death_date.isna(), 'last_alive_date'] = pd.NaT
    #test = df.loc[df.death_date.isna(), 'last_alive_date'].isna().sum()
    #print('\nNumber of records with last alive date or death date missing: \n{}'.format(test))

    # Retain gender, ethnicity, imdd, age, death date, last alive date
    df = df.rename(columns={'gender_code': 'gender', 'ethnicity_code': 'ethnicity'})
    df = df[['patient_id', 'gender', 'ethnicity', 'imdd_max', 'age_at_fit', 'death_date']]#, 'last_alive_date']]

    # Check missing values again
    test = df.isna().sum(axis=0)
    print('\nMissing value count after retaining one row per individual: \n{}'.format(test))

    # Quick explore
    print('\nGender:')
    print(df.groupby(['gender'])['patient_id'].nunique())
    print('\nEthnicity:')
    print(df.groupby(['ethnicity'])['patient_id'].nunique())
    print('\nAge at FIT test:')
    print(df.age_at_fit.describe())

    # Save data to disk?
    if save_data:
        df.to_csv(run_path / DEMO_OUT, index=False)

    return df
