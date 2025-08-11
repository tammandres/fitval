"""Prepare FIT values for the new FIT data product 
and extract some features (clinical symptoms, and whether FIT test done in stool pot) from free text.

Note: the FIT values come from 'fit_20240227.csv' provided by Brian Shine and this code cleans them,
and extracts features from free text associated with the FIT tests.

The FIT requests data comes from 'ice_reqs_20240423.csv', also provided by Brian Shine, and cleaned by Jaimie
The icen numbers in the two tables are anonymised together to enable linkage.
The FIT requests data is not processed in this script.
"""
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re
import warnings
from datetime import datetime
from textmining.symptoms import symptoms, add_symptom_indicators
from textmining.utils import extract


# Paths
print(os.getcwd())
lims_path = Path("<path redacted>")
out_path = Path('<path redacted>')
code_path = Path('<path redacted>')
if not code_path.exists():
    raise ValueError

for p in [lims_path, out_path, code_path]:
    print(p.exists())
    if not p.exists():
        p.mkdir(parents=True)

 # Initially this code was run without including symptoms, and later run with symptoms
include_symptoms = True  

# ---- Helper functions ----
#region        
def check_date_format(df, datecols, formats, regmap=None):
    """
    Checks if all dates in date columns of the dataframe conform to listed formats.
    Args:
      df : Pandas dataframe
      datecols : list of column names to be checked
      formats : list of date formats, e.g. ['%Y-%m', '%Y-%m-%d', '%Y-%m-%d %H:%M']
      valmap : dictionary {value:replacement} that allows to replace some values in date columns before these are checked
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
            warnings.warn("Some values do not conform to date formats")
            print('Values below do not conform to formats')
            print(s)


def clean_fit(fit):
    pats = [r'>', r'<', r'\.$', r'\*', r'\s', r'[a-zA-Z]', r'[\(\)\[\]]']
    for p in pats:
        matches = []
        for string in fit.dropna():
            out = re.search(p, string, flags=re.IGNORECASE)
            if out:
                matches.append(string)
        matches = list(np.unique(matches))
        print('\n--------')
        print('Pattern: {}'.format(p))
        print('Matches: {}'.format(matches))

    fit = fit.str.replace('|'.join(pats), '', regex=True)
    fit = fit.replace({'': np.nan})
    fit = fit.astype(float)
    return fit


#endregion

# ---- 1. Read data, reformat dates, clean FIT values, dbl check GP indicator ----
#region

# Read data, no nhsn for 401 records (112 gp)
str_dtypes = {col: 'str' for col in ['nhsn', 'mrn', 'dts', 'dtr', 'auth_dts', 'dob']}
df = pd.read_csv(lims_path / 'fit_20240227.csv', dtype=str_dtypes)
print('{} rows, {} columns'.format(df.shape[0], df.shape[1]))
print('\nColumns: {}'.format(df.columns.tolist()))
print('\ndtypes: \n{}'.format(df.dtypes))
print('\nmissingness: \n{}'.format(df.isna().sum()))

# Drop un-needed columns
df = df.drop(labels=['dob', 'sex'], axis=1)

# Drop duplicates
print('\n{} rows before dropping duplicates'.format(df.shape[0]))
df = df.drop_duplicates()
print('\n{} rows after dropping duplicates'.format(df.shape[0]))

# Remove last letter from mrn
check = df.mrn.str.contains('[a-zA-Z]$', regex=True).sum()
print('\nNumber of mrn numbers with last letter: {}'.format(check))
df.mrn = df.mrn.str.replace('[a-zA-Z]$', '', regex=True)
print('\nNumber of mrn numbers with last letter: {}'.format(df.mrn.str.contains('[a-zA-Z]$', regex=True).sum()))

# FIT dates
datecols = ['dts', 'dtr', 'auth_dts']
check_date_format(df, datecols, ['%Y-%m-%dT%H:%M:%SZ'])
for c in datecols:
    df[c] = pd.to_datetime(df[c], format='%Y-%m-%dT%H:%M:%SZ')

df = df.rename(columns={'dts':'fit_date', 'dtr': 'fit_date_received', 'auth_dts': 'fit_date_authorised',
                        'FIT_VALUE': 'fit_val', 'FIT_REPORT': 'fit_report'})

# According to Brian Shine, dts is date of sample, dtr is date of receipt; if date of sample not known they are equal.
# Check: date received is the same as fit date in about 27% of cases, o.w. larger
perc = [0.01, 0.05, 0.1, 0.25, 0.33, 0.5, 0.75, 0.9, 0.95, 0.99]
test = (df.fit_date_received - df.fit_date).dt.seconds / (3600)
test.describe(percentiles=perc)
(df.fit_date_received==df.fit_date).mean()

# Check: date authorised is greater than date received in 96% of cases, o.w. equal
test = (df.fit_date_authorised - df.fit_date_received).dt.seconds / (3600)
test.describe(percentiles=perc)
(df.fit_date_authorised > df.fit_date_received).mean()

# Clean FIT values
df['fit_val_clean'] = df.fit_val.copy()
df.fit_val_clean.str.contains(r'<\s*10', regex=True).sum()  # Dbl check
df['fit_val_clean'] = clean_fit(df.fit_val_clean)

# Dbl check unclean values where clean fit_val is nan
u = df.loc[df.fit_val_clean.isna(), 'fit_val']
print(len(u))
print('Unclean FIT values where clean FIT val is nan: {}'.format(u.unique()))

# Indicator for FIT >= 10
df['fit10'] = 0
df.loc[df.fit_val_clean >= 10, 'fit10'] = 1
print(df.fit10.mean() * 100)

# Check num tests per patient
print(df.mrn.nunique(), df.nhsn.nunique(), df[['mrn', 'nhsn']].isna().sum())
s = df.groupby('mrn').size().describe(percentiles=[0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
print(s)

# Check number of tests per year
print(df.groupby('fit_year').size())
print(df.fit_date.min(), df.fit_date.max())

# Glance
print(df.head())

# Check GP indicator
gp = pd.read_csv(code_path / 'gplocs20200526.txt', header=None)
gp.columns = ['loc']
df['gp'] = 0
df.loc[df['loc'].isin(gp['loc']), 'gp'] = 1
print(df.gp.mean())
test = df.gp == df.GP
print(all(test))
df = df.drop(labels=['GP'], axis=1)

# Dbl check if each nhsn associated with one mrn; plus missingness in nhsn and mnrn
test = df.dropna(subset=['nhsn']).groupby('nhsn')['mrn'].nunique().max()
print(test == 1)
print(df[['nhsn', 'mrn']].isna().sum())
print(df.mrn.nunique(), df.shape[0])

# Rename ICE number column
df = df.rename(columns={'ICE_n': 'icen'})

# Indicator for icen starting with 'ice'
df['request_number_starts_with_ice'] = 0
df.loc[df.icen.str.lower().str[:3] == 'ice', 'request_number_starts_with_ice'] = 1
df['request_number_missing'] = df.icen.isna().astype(int)

# <GP indicator based on both ICE number and location string>
df[['request_number_missing', 'request_number_starts_with_ice', 'gp']].value_counts(sort=False).reset_index()
df['gp_from_icen_and_loc'] = (df.gp == 1) | (df.request_number_starts_with_ice == 1)
df['gp_from_icen_and_loc'] = df['gp_from_icen_and_loc'].astype(int)
df[['gp_from_icen_and_loc', 'gp']].value_counts() 

#endregion


# ---- 2. Dbl check if any missing NHSn ----
#region

## < most of the code is redacted in this section as not essential to export > 
df['nhsn_mis'] = df.nhsn.isna()

#endregion


# ---- 3. Identify FIT tests done in stool pot ----
#region
pd.set_option('display.min_rows', 10, 'display.max_rows', 10, 'display.max_colwidth', 100)

# Create temporary, slightly cleaned comm1 field: all | are replaced with empty space, 
# all multiple spaces and newlines replaced with single space
df['comm1_tmp'] = df.comm1.fillna('').str.lower().str.replace(r'\|', ' ', regex=True).str.replace('r\s+', ' ', regex=True)

# .... 3.1. Explore all comments ....
explore = False
if explore:

    # Have a sense of how the comments are given
    c = df[['comm1']]
    c['length'] = c.comm1.fillna('').apply(len)
    c.length.describe()
    c = c.drop_duplicates().sort_values(by='length')
    print(c.shape)

    c.head(10)
    c.tail(10)
    #for __, row in c.iterrows():
    #    print('\n---')
    #    print(row.comm1)


    # Explore how comments are given for sample pickers and pots
    pat = r"\b(?:picker|device|sampler|collection|(?<=\W)pick\w*|collec\w*)"

    matches = extract(df, 'comm1_tmp', pat, pad_left=50, pad_right=50, flags=re.IGNORECASE|re.DOTALL)
    matches = matches.drop_duplicates(subset=['left', 'target', 'right']).sort_values(by=['left', 'target'])
    print(matches.shape)

    matches.target.str.lower().drop_duplicates().sort_values()

    for i, (j, row) in enumerate(matches.iterrows()):
        print(i, '|', row.left.lower(), '  <<  ', row.target.upper(), '  >>  ', row.right.lower())


    # Explore how comments are given for stool pots
    pat = r"stable in stool|pot returned|stool|stol|\bpot\b"

    matches = extract(df, 'comm1', pat, pad_left=20, pad_right=20, flags=re.IGNORECASE|re.DOTALL)
    matches = matches.drop_duplicates(subset=['left', 'target', 'right']).sort_values(by='target')
    print(matches.shape)
    matches.target.str.lower().drop_duplicates().sort_values()

    for i, (j, row) in enumerate(matches.iterrows()):
        print(i, '|', row.left.lower(), row.target.upper(), row.right.lower())

# .... 3.2. Identify relevant comments ....

# Stool pot indicator: main
pat = r"sample not received in fit collection device"
matches = extract(df, 'comm1_tmp', pat, pad_left=50, pad_right=50, flags=re.IGNORECASE|re.DOTALL)
matches.target.value_counts()  # 7199
matches.row.nunique()  # 7197

matches = matches.drop_duplicates(subset=['left', 'target', 'right'])
matches.shape 
matches.target.str.lower().drop_duplicates()

mask = df.comm1_tmp.str.contains(pat, regex=True)
df['stool_pot'] = 0
df.loc[mask, 'stool_pot'] = 1
print(df.stool_pot.mean(), df.stool_pot.sum())
df.loc[df.stool_pot==1, 'comm1'].head()

# Stool pot indicator: wider definition
pats = [r"\bnot (?:receive|arrive)[^\.]{,40} in(?:side)?[^\.]{,20}(?:collect|devic|pick)\w*",
        r"\bnot in(?:side)?[^\.]{,20}(?:collect|devic|pick)\w",
        r"(?:receive|arrive)[^\.]{,20} in(?:side)?[^\.]{,20}bag",
        r"outside[^\.]{,20}(?:collect|devic|pick)\w",
        r"(?:collect|devic|pick)\w[^\.]{,20}unused"
        ]
pat = '|'.join(pats)

matches = extract(df, 'comm1_tmp', pat, pad_left=50, pad_right=50, flags=re.IGNORECASE|re.DOTALL)
matches.target.value_counts()  # vast majority is stool_pot indicator comment (7199)
matches.target.value_counts().iloc[1:].sum()  # 82 other examples, e.g. received in bag
for t in matches.target.unique():
    print(t)

for i, (j, row) in enumerate(matches.iterrows()):
    print(i, '|', row.left.lower(), '<<', row.target.upper(), '>>', row.right.lower())

mask = df.comm1_tmp.str.contains(pat, regex=True)
df['stool_pot2'] = 0
df.loc[mask, 'stool_pot2'] = 1
print(df.stool_pot2.mean(), df.stool_pot2.sum())
df.loc[df.stool_pot2==1, 'comm1'].drop_duplicates().head()

df[['stool_pot', 'stool_pot2']].value_counts()  # 77 extra matches with additional patterns
#endregion


# ---- 4. Reformat and save ----
# This is only run if symptoms are not included. In that case, free text dropped and data saved
#region 
if not include_symptoms:

    # Drop free text and nhsn mis indicator
    print(df.columns)
    df = df.drop(labels=['clind', 'comm1', 'comm1_tmp', 'nhsn_mis'], axis=1)
    print(df.columns)

    # Unique values - dbl check to ensure no text leaked into fields
    for c in df.columns:
        if c != 'nhsn' and c != 'mrn':
            print(c, '|', df[c].unique())

    # Check data types
    df.dtypes
    df[[c for c in df.columns if 'date' in c]].dtypes

    # Check ice
    df.icen.dropna().str.contains('^ICE\-\d{9}$|^\d{9}$', regex=True).mean() # all icen same format

    # Check fit_val
    test = df.fit_val.astype(str).apply(len)
    df.fit_val[test > 6].str.replace('\[|\]', '').unique()

    # Visual check
    df.head()

    # Save
    df.to_csv(out_path / 'fit_values_20240612.csv', index=False)

#endregion


# ---- 5. Identify clinical symptoms and save ----
# This is run if symptoms are included
#region
if include_symptoms:

    # Extract symptoms, using mrn number as the dummy subject identifier
    run_path = Path(os.getcwd())
    df['subject'] = df.mrn.copy()

    run_sym = False
    if run_sym:
        __, matches = symptoms(run_path, df, col='clind', save_data=False)
        matches.to_csv(out_path / 'matches_symptoms_20240612.csv', index=False)
    else:
        matches = pd.read_csv(out_path / 'matches_symptoms_20240612.csv')

    matches.loc[matches.exclusion_indicator==0].category.value_counts()
    matches['phrase'] = matches.left + matches.target + matches.right

    # Examine matches 
    matches_incl = matches.loc[matches.exclusion_indicator == 0]
    matches_incl.shape
    t = matches_incl.drop_duplicates(subset='target')
    t = t.sort_values(by=['category', 'target'])
    t.shape
    for i, (j, row) in enumerate(t.iterrows()):
        print(i, j, row.category, row.target.upper(), '|', row.left.lower(), '<TARGET>', row.right.lower())

    # Inspect where keyword 'reduced' is on left side, as previously in old code these were excluded
    m = matches.loc[matches.left.str.lower().str.contains('reduce')]
    for i, (j, row) in enumerate(m.iterrows()):
        print(i, j, row.category, row.target.upper(), '|', row.left.lower(), '<TARGET>', row.right.lower())

    # Additional patterns to exclude from automatically included matches
    pats = {'abdopain': 'exclude abd|pain/concerned bowel|ache ssettle',
            'bloodsympt': 'menst.{,20}bleed',
            }

    matches_incl_cleaned = matches_incl.copy()
    for cat, pat in pats.items():
        print('\n----', cat)
        submatches = matches_incl.loc[matches_incl.category==cat]
        mask = submatches.phrase.str.lower().str.contains(pat, regex=True, flags=re.I|re.DOTALL)
        submatches = submatches.loc[mask]
        print(mask.sum())
        print(submatches.phrase.unique())
        matches_incl_cleaned = matches_incl_cleaned.loc[~matches_incl_cleaned.index.isin(submatches.index)]

    matches_incl.shape
    matches_incl_cleaned.shape

    # Examine excluded matches 
    matches_excl = matches.loc[matches.exclusion_indicator == 1]
    matches_excl.shape

    mask = matches_excl.left.str.lower().str.contains(r'\b(?:not|no|no visible|nil|no obvious|no \w+ or)\s*$', regex=True)
    mask.sum()
    m0 = matches_excl.loc[mask]  # patterns 'no <TARGET>'
    m1 = matches_excl.loc[~mask]  # other patenrs

    for i, (j, row) in enumerate(m0.iterrows()):
        print(i, j, row.category, row.target.upper(), '|', row.left.lower(), '<TARGET>', row.right.lower())

    for i, (j, row) in enumerate(m1.iterrows()):
        print(i, j, row.category, row.target.upper(), '|', row.left.lower(), '<TARGET>', row.right.lower())

    # Additional patterns to include from excluded matches
    #  < These patterns are not included in exported code as they contain short strings from clinical details >
    #  and not essential to be included 
    #pats = {'abdopain': <redacted>,
    #        'abdomass': <redacted>,
    #        'ida': <redacted>,
    #        'anaemia': <redacted>,
    #        'rectalpain': <redacted>,
    #        'rectalbleed': <redacted>,
    #        'bloat': <redacted>,
    #        'bowelhabit': <redacted>,
    #        'fh': <redacted>,
    #        'constipation': <redacted>,
    #        'diarr': <redacted>,
    #        'bloodsympt': <redacted>,
    #        }

    matches_add = pd.DataFrame()
    for cat, pat in pats.items():
        print('\n-------', cat)
        submatches = matches_excl.loc[matches_excl.category == cat]
        mask = submatches.phrase.str.lower().str.contains(pat, regex=True, flags=re.I|re.DOTALL)
        if mask.sum() > 0:
            print(mask.sum())
            m = submatches.loc[mask]
            print(m.phrase.unique())
            matches_add = pd.concat(objs=[matches_add, m], axis=0)
    
    matches_add
    matches_add['exclusion_indicator'] = 0
    matches_add['exclusion_reason'] = ''

    matches_incl_cleaned.shape
    matches_incl_cleaned = pd.concat(objs=[matches_incl_cleaned, matches_add], axis=0)
    matches_incl_cleaned.shape
    matches_incl.shape

    matches_incl_cleaned.to_csv(out_path / 'matches_symptoms_included_cleaned_20240612.csv', index=False)

    # Add symptom indicators to FIT dataframe
    df.shape
    df = add_symptom_indicators(df, matches_incl_cleaned)
    df.shape

    # Note: in some cases, still ida and anaemia coexist. May need to adjust later, e.g. set anaemia to 0 in that case.
    # Corresponds to cases like 'anaemia - iron def; anaemia iron def, 'anaemia with iron def', 'low iron, low hb' etc.
    df[['symptom_ida', 'symptom_low_iron', 'symptom_anaemia']].value_counts(sort=False).reset_index()
    df.loc[(df.symptom_ida==1)&(df.symptom_anaemia==1), 'clind']
    m = df.loc[(df.symptom_ida==0)&(df.symptom_low_iron==1)&(df.symptom_anaemia==1), 'clind']
    m.unique()

    # Drop free text and nhsn mis indicator
    print(df.columns)
    df = df.drop(labels=['clind', 'comm1', 'comm1_tmp', 'nhsn_mis'], axis=1)
    print(df.columns)
    df = df.drop(labels=['subject'], axis=1)
    df.columns
    df = df.drop(labels=['row'], axis=1)
    df.columns

    # Unique values - dbl check to ensure no text leaked into fields
    for c in df.columns:
        if c != 'nhsn' and c != 'mrn':
            print(c, '|', df[c].unique())
    # Save
    df.to_csv(out_path / 'fit_values_with_symptoms_20240612.csv', index=False)

#endregion