"""Helper functions for data quality checks"""
import pandas as pd  
import numpy as np 
import regex as re   # For regular expressions   
import time
import warnings
from datetime import datetime
import json
warnings.simplefilter('always', UserWarning)


def extract(df, col='safe_imaging_report', pat='pattern', pad_left=100, pad_right=100,
            flags=0, groups=True):
    """
    Get matches for pattern 'pat' in column 'col' in dataframe 'df'
    If 'pat' contains capture groups, the output contains 'target_group1', 'target_group2' etc

    Output:
      Dataframe with columns:
      'row' : row number of 'df' that contains the match (NB -- NOT necessarily the index of a Pandas dataframe)
      'start' : start position of a match in string
      'end' : end position of a match in string
      'left' :  pad_left characters from the left of match
      'right' : pad_right characters from the right of match
      'target' : target
    """
    # Get data
    reports = df[col].copy()

    # Convert to string
    reports = reports.astype(str)

    # Replace NAN with ''
    if reports.isna().sum() > 0:
        warnings.warn("Some input rows are NaN, replacing with ''")
        reports[reports.isna()] = ''

    # Detect number of capture groups
    n_group = re.compile(pat).groups

    # Create an empty data frame for storing results
    if groups:
        target_cols = ['target'] + ['target_group' + str(x) for x in range(1, n_group+1)]
    else:
        target_cols = ['target']
    colnames = ['row', 'start', 'end', 'left'] + target_cols + ['right']
    matches = pd.DataFrame(columns=colnames)

    # Loop over reports
    for i, txt in enumerate(reports):
        if txt:  # Proceed if the row is not empty
            match = re.search(pat, txt, flags=flags)
            if match:  # Proceed if any matches were found
                match = re.finditer(pat, txt, flags=flags)
                for m in match:  # Loop over matches within report
                    start = m.start()
                    end = m.end()
                    target = m.group(0)

                    left = start - pad_left
                    right = end + pad_right
                    if left < 0:
                        left = 0
                    if right > len(txt):
                        right = len(txt)
                    left = txt[left:start]
                    right = txt[end:right]

                    if groups:
                        target = ['' if not m.group(i) else m.group(i) for i in range(n_group+1)]  # Get matches for all capture groups
                    else:
                        target = [target]

                    data = [i, start, end, left] + target + [right]  # Gather all extracted data
                    s = pd.DataFrame(data).transpose()
                    s.columns=colnames
                    matches = pd.concat(objs=[matches, s], axis=0)

    # Tidy up
    matches = matches.astype({'row': 'int32', 'start':'int32', 'end':'int32'})
    matches = matches.reset_index(drop=True).copy()
    return matches


def check_empty(file):
    """
    Checks if all cells in a Jupyter nb are empty
    """
    f = open(file, encoding='utf8')
    book = json.load(f)
    f.close()
    count = 0
    for c in book['cells']:
        if 'outputs' in c.keys():
            if c['outputs']:
                count += 1
    return count


def explore_encoding(files, encoding='utf-8', nrows=None, nchunk=10, sep=None):
    """
    Reads each csv file in files using a prespecified encoding and displays the result
    If this results in error, it reads the file but replaces all encoding errors with unicode replacement char '\uFFFD'
    It will then print the number of encoding errors as well as extracts of text where these occurred
    
    Args
     files : dict {filename : encoding} or list of filenames. If list of filenames, same encoding (in 'encoding') is tried for all files
     encoding : encoding to try on all files; only used if files is list and not dict
    """
    if type(files) is dict:
        files, encodings = list(files.keys()), list(files.values())
    else:
        encodings = [encoding]*len(files)
    res = pd.DataFrame()
    err = pd.DataFrame()
    for f, e in zip(files, encodings):
        print('\nReading table {}, using encoding {}'.format(f, e))
        count = 0
        success = 0
        try:
            test = pd.read_csv(f, encoding=e, sep=sep, engine='python', nrows=nrows, encoding_errors='strict')
            success = 1
            print('Table read successfully.')
            print(test.head())
        except Exception as ex:
            print("! {}: {}".format(ex.__class__.__name__, ex))
            print("! Setting encoding_errors='replace'")
            try:
                test = pd.read_csv(f, encoding=e, sep=sep, engine='python', nrows=nrows, encoding_errors='replace')
                success = 1
                
                # Count and extract encoding errors
                errors = pd.DataFrame()
                for col in test.columns:
                    stop = 0
                    for chunk in np.array_split(test[[col]], nchunk):
                        m = extract(chunk, col=col, pat='\uFFFD', pad_left=20, pad_right=20)
                        m['column'] = col
                        count += len(m)
                        errors = pd.concat(objs=[errors, m], axis=0)
                        if count > 100:
                            print("! Number of encoding errors is at least 100, not counting further errors".format(count))
                            stop = 1
                            break
                    if stop == 1:
                        break
                errors['file'] = f
                err = pd.concat(objs=[err, errors], axis=0)
                print("! Number of encoding errors: {}".format(count))
                print("! Snapshot of encoding errors:")
                print(errors)
                print("! Snapshot of output:")
                print(test.head())
                """
                count = 0
                for col in test.columns:
                    s = ''.join(test[col].astype(str))
                    m = re.findall('\uFFFD', s)
                    count += len(m)
                    if count > 1000:
                        print("! Number of encoding errors is at least 1000")
                        break
                print("! Number of encoding errors is: {}".format(count))
                """
            except Exception as ex:
                print("! {}: {}".format(ex.__class__.__name__, ex))
                print("! Unable to read file. Terminating.")
                break
        
        # Check for unusual characters in column names
        colnames = test.columns.to_list()
        print('Inspect column names for unusual characters: \n{}'.format(colnames))
        tmp = repr(colnames)
        tmp = pd.DataFrame([tmp], columns=['col'])
        check = extract(tmp, 'col', r'\\u')
        if check.shape[0] > 0:
            print('! Columns contain unusual characters')
            print(check)

        # Store
        line = pd.DataFrame([[f, e, count, success]], columns=['table', 'attempted_encoding', 'error_count', 'file_read_successfully'])
        res = pd.concat(objs=[res, line], axis=0)
    return res, err


def check_identifiers(tables, tref='', subjcols=['subject', 'subject_id', 'salted_master_patient_id', 'patient_id'],
                      encoding='utf-8', engine='python', sep=None, testmode=False):
    """
    For each table in 'tables', checks whether subject identifiers are present in a reference table
    Returns a dataframe containing table names and a new boolean column 'identifiers_match'
    
    Args
      tables : a list of filenames (csv), or a dict {filename:encoding}
      tref : filename of the reference table, or a dict {filename:encoding} for a reference table
      subjcols : possible names (in lowercase) for how a patient id column may be named
      encoding : encoding to be applied to all tables, unless tables and tref are dicts
      testmode : if True, reads only first 10 rows of each table
    
    Notes
      If sep=None and engine='python', separator is detected automatically
    """
    
    print('\nChecking if subject identifiers match...')
    tic = time.time()
    nrows = 10 if testmode else None

    # Read reference table
    ref_name = list(tref.keys())[0] if type(tref) is dict else tref
    ref_encoding = list(tref.values())[0] if type(tref) is dict else encoding
    try:
        print('Reading reference table {}, using encoding {}'.format(ref_name, ref_encoding))
        dfref = pd.read_csv(ref_name, sep=sep, engine=engine, encoding=ref_encoding, encoding_errors='strict',
                            usecols=lambda x: x.lower() in subjcols, nrows=nrows)
    except Exception as ex:
        # Try reading with replacing encoding errors
        print("! {}: {}".format(ex.__class__.__name__, ex))
        print("! Potential encoding error. Setting encoding_errors='replace'")
        dfref = pd.read_csv(ref_name, sep=sep, engine=engine, encoding=ref_encoding, encoding_errors='replace',
                            usecols=lambda x: x.lower() in subjcols, nrows=nrows)
        print("  Snapshot of dataframe when encoding errors were replaced:\n{}".format(dfref.head()))
    dfref.columns = ['subject']
    dfref = dfref.drop_duplicates()   
        
    # Get names and encodings of target tables
    target_names = list(tables.keys()) if type(tables) is dict else tables
    target_encodings = list(tables.values()) if type(tables) is dict else [encoding]*len(tables)
    #target_names = [t for t in target_names if t != ref_name]
    #target_encodings = [e for e,t in zip(target_encodings, target_names) if t != ref_name]
    
    # Compare subject identifiers in other tables to reference table
    s = pd.DataFrame(target_names, columns=['file']) 
    s['identifiers_match'] = np.nan
    for target, encoding in zip(target_names, target_encodings):
        try:
            print('Reading {}, using encoding {}...'.format(target, encoding))
            try:
                df = pd.read_csv(target, sep=sep, engine=engine, encoding=encoding, encoding_errors='strict',
                                 usecols=lambda x: x.lower() in subjcols, nrows=nrows)
            except Exception as ex:
                print("! {}: {}".format(ex.__class__.__name__, ex))
                print("! Potential encoding error. Setting encoding_errors='replace'")
                df = pd.read_csv(target, sep=sep, engine=engine, encoding=encoding, encoding_errors='replace',
                                 usecols=lambda x: x.lower() in subjcols, nrows=nrows)
                print("  Snapshot of dataframe when encoding errors were replaced:\n{}".format(df.head()))
            df.columns = ['subject']
            df = df.drop_duplicates()

            # Test if all subjects in target table are present in reference table
            test = df.subject.isin(dfref.subject)
            nchar = 50
            if test.all() == 0:
                print('! {}/{} subjects in this table are in reference table'.format(test.sum(), len(test)))
                s.loc[s.file == target, 'identifiers_match'] = False
            else:
                print('  {}/{} subjects in this table are in reference table'.format(test.sum(), len(test)))
                s.loc[s.file == target, 'identifiers_match'] = True
        except:
            df = pd.read_csv(target, sep=sep, engine=engine, encoding=encoding, encoding_errors='replace', nrows=2)
            test = df.columns.isin(subjcols).sum()
            if test == 0:
                print("\n! This table does not contain patient identifiers with names {}".format(subjcols))
    toc = time.time()
    print('Time elapsed: {:.2f} seconds'.format(toc-tic))
    return s


def describe_tables(tables, encoding='utf-8', subjcols=['subject', 'subject_id', 'salted_master_patient_id', 'patient_id'],
                    sep=None, engine='python', testmode=False):
    """
    For each table in 'tables', extracts the number of rows and columns, number of patients, and column names
    Returns these in a dataframe
    
    Args
     tables : list of filenames (e.g. csv files) or list of table names in an SQL database
     sep : separator to be used when reading tables from file
     testmode : if True, reads only first 10 rows of each table
    """
    if type(tables) is dict:
        tables, encodings = list(tables.keys()), list(tables.values())
    else:
        encodings = [encoding]*len(tables)
    nrows = 10 if testmode else None

    print('\nSummarising tables...')
    tic = time.time()
    s = pd.DataFrame(columns=['file', 'n_row', 'n_col', 'n_patient', 'columns'])
    for t, e in zip(tables, encodings):

        # Read table
        print('   Reading table {}, using encoding {}'.format(t, e))
        try:
            df = pd.read_csv(t, sep=sep, engine=engine, encoding=e, encoding_errors='strict', nrows=nrows)
        except Exception as ex:
            print("! {}: {}".format(ex.__class__.__name__, ex))
            print("! Potential encoding error. Setting encoding_errors='replace'")
            df = pd.read_csv(t, sep=sep, engine=engine, encoding=e, encoding_errors='replace', nrows=nrows)
            print("  Snapshot of dataframe when encoding errors were replaced:")
            print(df.head())

        # Compute number of subjects, if present
        cols = df.columns
        cols_low = [c.lower() for c in cols]
        test = np.isin(cols_low, subjcols)
        nsub = df[cols[test][0]].nunique() if test.any() else np.nan
        
        # Summarise the table and append to results container
        df_sum = pd.DataFrame([[t, df.shape[0], df.shape[1], nsub, df.columns.to_numpy()]], columns=s.columns)
        s = pd.concat(objs=[s, df_sum], axis=0).reset_index(drop=True)
        del df
    toc = time.time()
    print('Time elapsed: {:.2f} seconds'.format(toc-tic))
    return s


def to_datetime_multi(df, datecols, formats):
    """
    Converts datestrings to Pandas datetime
    
    Assumes that dataframe 'df' contains datestrings in columns listed in 'datecols'
    The list of formats in 'formats' is used to convert all datestrings to common datetime format
    Useful when a single column contains datestrings in multiple formats that have the same time sequence
    (e.g. dd-mm-yyyy, dd-mm-yy, d/m/y)
    NB: this code will lead to wrong results if time sequences are different,
        e.g. if '10/05/2021' means day=5, month=10 in one row, and day=10 month = 5 in another
    
    Args : 
      df : Pandas dataframe
      datecols : list of column names that contain datestrings
      formats : list of date formats that are used to convert datestrings to dates
    """
    for c in datecols:
        #df[c] = pd.to_datetime(df[c], dayfirst=dayfirst)
        datemask = ~df[c].isna()
        dates = pd.to_datetime(df[c].copy(), format=formats[0], errors='coerce')  # Convert using first format in formats
        #print(dates)
        if len(formats) > 1: # If more formats were given, apply these to yet unconveted rows
            for f in formats[1:]:  # Try format f on dates that have not been converted so far
                mask = dates.isna() & datemask
                if np.any(mask):
                    dates[mask] = pd.to_datetime(df[c][mask].copy(), format=f, errors='coerce')
                #print(dates)
        test = np.any(dates[datemask].isna())
        if test:
            #warnings.warn('Not all dates were converted')
            raise ValueError('Not all dates were converted')
        df[c] = dates 
    return df


def check_date_format(df, datecols, formats, regmap=None):
    """
    Checks if all dates in date columns of the dataframe conform to listed formats.
    Args:
      df : Pandas dataframe
      datecols : list of column names to be checked
      formats : list of date formats, e.g. ['%Y-%m', '%Y-%m-%d', '%Y-%m-%d %H:%M']
      regmap : dictionary {value:replacement} that allows to replace some values in date columns before these are checked
    Idea is based on user cs95 @https://stackoverflow.com/questions/49435438/pandas-validate-date-format
    Format codes: https://docs.python.org/3/library/datetime.html
    
    Compared to older version: changes dict name to regmap to be consistent with change_format() function
    """
    for c in datecols:
        print('\nColumn: {}'.format(c))
        s = df[c].drop_duplicates().copy()
        if regmap is not None:
            if c in regmap:
                s = s.replace(regmap[c], regex=True)
        mask = np.zeros(s.shape, dtype=bool)
        for f in formats:
            for i, datestring in enumerate(s):
                try:
                    datetime.strptime(datestring, f)
                    mask[i] = True
                except:
                    pass
        s = s[~mask].dropna().sort_values()
        if s.empty:
            print('All values conform to formats')
        else:
            #warnings.warn("Some values do not conform to date formats")
            #print('Values below do not conform to formats')
            print(s)
            raise ValueError("Values above do not conform to formats.")


def get_ethnic_dict():
    # Based on #https://datadictionary.nhs.uk/data_elements/ethnic_category.html
    ethnic_dict = {
        'A': 'British',
        'B': 'Irish',
        'C': 'Any other White background',
        'D': 'White and Black Caribbean',
        'E': 'White and Black African',
        'F': 'White and Asian',
        'G': 'Any other mixed background',
        'H': 'Indian',
        'J': 'Pakistani',
        'K': 'Bangladeshi',
        'L': 'Any other Asian background',
        'M': 'Caribbean',
        'N': 'African',
        'P': 'Any other Black background',
        'R': 'Chinese',
        'S': 'Any other ethnic group',
        'Z': 'Not stated',
        '99': 'Not known'
    }
    return ethnic_dict
