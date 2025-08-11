"""Extract colorectal cancer (present/absent) and its site from pathology reports

The extraction is done in two steps: 
(1) All matches for colorectal tumour keywords are automatically extracted 
    using the get_crc_reports_par function from textmining.reports in line 94.

    This produces a table that has all extracted tumour keywords with left and right context,
    and an 'exclusion_indicator' column that shows which were automatically excluded 
    (e.g. because these were not associated with colorectal site).
    The column 'row' indicates the row of the reports table from which the match was found.

(2) The matches are then semi-manually processed to remove false positives.
    This step is very laborious and takes up most of code space in this script.
    Without it, the result would be pretty good, although still around 8% of matches can be false positives.

NB this code will likely not work on a different set of clinical reports.
The regex patterns that were used in step (2) to retain or keep the matches for tumour keywords 
wre specifically designed to work on the set of reports that was analysed. 
The validity of these regex patterns was checked by printing out matches to these regex patterns
(the printouts are not included in this script).
"""
from pathlib import Path
import pandas as pd
import warnings
from datetime import datetime
import numpy as np
from textmining.reports import get_crc_reports_par
from textmining.utils import pat_from_vocab, constrain_distance
from textmining.constants import VOCAB_DIR
import regex as re
from textmining.utils import extract


out_path = Path('<path redacted>')
data_path = Path('<path redacted>')


## ==== I. Read data and extract matches for tumour keywords and anatomical sites ====
## using a regex-based algorithm (get_crc_reports_par) that is designed to identify phrases which discuss current colorectal cancer
#region
datecol = 'received_date'
report_col = 'unsafe_report' 

# Get reports
df = pd.read_csv(data_path, sep=',', engine='c', lineterminator='\n')
print('\nNumber of pathology reports: ' + str(df.shape[0]))
print('\nColumns: {}'.format(df.columns))
print('\nMissingness: \n{}'.format(df.isna().sum(axis=0)))

# Retain only some columns
df = df[['nhs_number', 'mrn_number', 'unsafe_imaging_report', 'sent_date', 'received_date', 'authorised_date',
         'date_offset_days', 'snomed_t', 'snomed_m']]
df = df.rename(columns={'unsafe_imaging_report': 'unsafe_report'})

# Date to datetime
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

datecols = ['sent_date', 'received_date', 'authorised_date']
check_date_format(df, datecols=datecols, formats=['%Y-%m-%d'])
for c in datecols:
    df[c] = pd.to_datetime(df[c], format='%Y-%m-%d')

# Run text extraction
run_nlp = False
if run_nlp:
    #_, matches = get_crc_reports(df, report_col, verbose=False, add_subj_to_matches=False, subjcol='subject', negation_bugfix=True)
    matches = get_crc_reports_par(nchunks=10, njobs=4, df=df, col=report_col, add_subj_to_matches=False, negation_bugfix=True)
    matches.to_csv(out_path / 'unsafe_matches_crc_20240617.csv')
else:
    matches = pd.read_csv(out_path / 'unsafe_matches_crc_20240617.csv')
#endregion


## ---- Get regex patterns for colorectal anatomical sites
#region
VOCAB_FILE = 'vocab_site_and_tumour.csv'
v = pd.read_csv(VOCAB_DIR / VOCAB_FILE)
vsite = v.loc[v.cui.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
print('\nSites included: {}'.format(vsite.concept.unique()))
pats_site = pat_from_vocab(vsite, gap=r'\s{1,3}', verbose=False, name='sites', spellcheck=True)
#site_right = constrain_distance(pats_site, side='right', char='.', distance=20)

vsite.loc[vsite.pat.str.contains('rect.{,10}sigm', regex=True, flags=re.DOTALL), 'cui'] = 100
vsite.loc[vsite.pat.str.contains('rect.{,10}sigm', regex=True, flags=re.DOTALL), 'concept'] = 'rectosigmoid'
vsite.concept = vsite.concept.replace({'colon and rectum': 'unknown'})
pats_site_dict = {}
for cui in vsite.cui:
    vsite_sub = vsite.loc[vsite.cui==cui]
    pats_site_sub = pat_from_vocab(vsite_sub, gap=r'\s{1,3}', verbose=False, name='sites', spellcheck=True)
    concept = vsite_sub.concept.iloc[0]
    pats_site_dict[concept] = pats_site_sub

# Get patterns for excluded sites
sites_ex = ['liver', 'lung', 'uterus',
            'ovaries', 'bladder',
            'spleen', 'anastomosis', 'adrenal gland', 'kidney',
            'bone', 'pleura', 'brain', 'head', 'prostate'] 
vsite_ex = v.loc[v.concept.isin(sites_ex)]
ex_site = pat_from_vocab(vsite_ex, gap=r'\s{1,3}', verbose=False, name='sites', spellcheck=True)
ex_site = ex_site + '|appendic'
site_ex_left = constrain_distance(ex_site, side='left', char='.', distance=25)
site_ex_right = constrain_distance(ex_site, side='right', char='.', distance=25)


#endregion



## ==== II. Separate out high confidence matches ====

# Helper functions
#region
def find_first(df: pd.DataFrame, col: str, pat: str):
    if df.index.shape[0] != df.index.nunique():
        raise ValueError('index of df must be unique')
    m = df[col].str.lower().str.findall(pat, flags=re.DOTALL|re.I)
    m = m.apply(lambda x: x[0] if x else np.nan).rename(col)
    dfsub = df.drop(labels=[col], axis=1).merge(m, left_index=True, right_index=True)
    dfsub = dfsub.dropna(subset=[col])
    return dfsub


def find_first_v2(df, col, pat):
    if df.index.shape[0] != df.index.nunique():
        raise ValueError('index of df must be unique')
    m = extract(df, col, pat, flags=re.I)
    m = m.sort_values(by=['row', 'start']).groupby(['row']).first().reset_index()
    idx_matches = df.iloc[m.row].index
    dfsub = df.loc[idx_matches].copy()
    dfsub.loc[idx_matches, col] = m.target.values
    return dfsub


def clean(df, cols, pat=r'\r+|\n+', repl=' <n> '):
    for col in cols:
        df[col] = df[col].str.replace(pat, repl, regex=True).str.lower()
    return df


def dummy_redact(r: str):

    start0 = r'\b(?:drs?|prof|professor|mrs|ms)\b\.?'  # dr, dr., mrs, mrs., ms, ms.
    start1 = r'\bmr\b\.?(?!\s(?:scan|img|image|imaging|pelvi|contra|lumbo|lumba|head|thora|spine|chest|neck|exam|invest))'  # "mr" but not "mr scan"
    start2 = r'\b(?:addendum start|dissected|cut|chaperoned|supervi[sz]ed|supervision|review|reviewed|written|reported|authori[sz]ed) by[ \t]*[:,\-]?'
    start3 = r'\b(?:consultants?|trainees?|pathologist|authori[sz]ed|signed|reviewed)[ \t]*[:\-]'  # Not optional : or -, otherwise more false pos 
    start4 = r'\b(?:surnames?|forenames?|names?|named as|named|name changed from)[ \t]*[:,\-]?'  # Optional ':', ',' or '-'
    excl = r'(?![ \t](?:and|or|with|from|where|drs?|prof|professor|mrs|ms)\b)'  # Assuming none of these are names
    end = r"[ \t]+'?[a-z'`]+(?:[ \t\-\.,]{1,2}[a-z'`]+){,3}\b"  # Do not incl \d in allowed char, otherwise could rm TNM staging, e.g. mr T0N0
    name_regex = '(?:' + '|'.join([start0, start1, start2, start3, start4]) + ')' + excl + end

    phone_regex = r"(?:" + r"(?:(?:(?:\+|00)\d{1,3})|[0])" + r"\d{3}\s*\d\s*\d{2}\s*\d\s*\d{3}" + r")"

    nhs = r"(?:\d{3}[ -]?\d{3}[ -]?\d{4})"
    nhs2 = r"(?:\d{3}[ -]\d{4}[ -]\d{3})"
    mrn = r"(?:RTH\d+)"
    patient_regex = '|'.join([nhs, nhs2, mrn])

    pats = {'<nhsn_or_mrn>': patient_regex,
            '<phone>': phone_regex,
            '<name>': name_regex}

    for name, pat in pats.items():
        r = re.sub(pat, name, r, flags=re.IGNORECASE|re.DOTALL)
    
    return r


#endregion



# ---- 0. Get relevant matches for tumour keywords according to the algorithm and remove some obvious false positives ----
#region
if 'Unnamed: 0' in matches.columns:
    matches = matches.rename(columns={'Unnamed: 0': 'idx'})
matches.shape[0] == matches.idx.nunique()

matches_incl = matches.loc[matches.exclusion_indicator == 0].copy()
matches_excl = matches.loc[matches.exclusion_indicator == 1].copy()
matches_incl.shape
matches_excl.shape
                           
for col in ['left', 'target', 'right']:
    matches_incl[col + '_proc'] = matches_incl[col].copy()
    matches_excl[col + '_proc'] = matches_excl[col].copy()

matches_conf = pd.DataFrame()

df['row'] = np.arange(df.shape[0])
df['crc_nlp_orig'] = 0
df.loc[df.row.isin(matches_incl.row), 'crc_nlp_orig'] = 1


def remove_fp(data, col, pat, preview=False, rm_report=False, pat_keep=None):

    if 'tmp_exclusion' in data.columns:
        print('Removing tmp_exclusion column')
        data = data.drop(labels=['tmp_exclusion'], axis=1)

    data['tmp_exclusion'] = 0
    data.loc[data[col].str.lower().str.contains(pat, flags=re.DOTALL), 'tmp_exclusion'] = 1
    nmatch = data.tmp_exclusion.sum()
    print('Number of matches:', nmatch)

    if pat_keep is not None:
        mask = data[col].str.lower().str.contains(pat_keep, flags=re.DOTALL)
        mask = mask & (data.tmp_exclusion == 1)
        data.loc[mask, 'tmp_exclusion'] = 0
        print('\nKept phrases:')
        phrases_k = data.loc[mask, col].sort_values().str.replace('\r|\n', ' <n> ').unique()
        for i, c in enumerate(phrases_k):
            print(i, c)

    if nmatch > 0:
        print('\nRemoved phrases')
        phrases = data.loc[data.tmp_exclusion==1, col].sort_values().str.replace('\r|\n', ' <n> ').unique()
        for i, c in enumerate(phrases):
            print(i, c)
        
        if not preview:
            print('Removing matches...')
            print(data.shape[0])
            row_ex = data.loc[data.tmp_exclusion==1].row
            if rm_report:
                data = data.loc[~data.row.isin(row_ex)]
            else:
                data = data.loc[data.tmp_exclusion==0]
            print(data.shape[0])
    return data


# Remove some false positive matches

## Links
matches_incl = remove_fp(matches_incl, 'left', r'\/.{,25}\/.{,25}\/.{,25}$', rm_report=False)  ## 
matches_incl = remove_fp(matches_incl, 'right', r'^.{,25}\/.{,25}\/.{,25}\/.{,25}', rm_report=False)  ## 

## Recurrence
matches_incl = remove_fp(matches_incl, 'left', r'\brecur|\bregrow', rm_report=True, preview=False)  ## 
matches_incl = remove_fp(matches_incl, 'right', r'\brecur|\bregrow', rm_report=True, preview=False)  ## 

## Spread
matches_incl = remove_fp(matches_incl, 'left', r'spread from (\w+ ?){1,2} (?:colorect|rect|appendi|primary)|spread from(?: a ?)?$', 
                         rm_report=True, preview=False)  # 
matches_incl = remove_fp(matches_incl, 'right', r'spread from (\w+ ?){1,2} (?:colorect|rect|appendi|primary)', 
                         rm_report=True, preview=False)  ## 

## small bowel - more obvious matches
matches_incl = remove_fp(matches_incl, 'left', r'\b(?:biops|site)[^\n\r]{,20}(?:small bowel|ileum|ileal)[^\n\r]{,50}$', rm_report=True, preview=False)  ##  
matches_incl = remove_fp(matches_incl, 'left', r'site of tumour[^\n\r]{,20}(?:small bowel|ileum|ileal)', rm_report=True, preview=False)  ##  
matches_incl = remove_fp(matches_incl, 'left', r'(?:small bowel|ileum|ileal)[^\n\r]{,20}biops[^\n\r]{,50}$', rm_report=True, preview=False)  ##

## Too superficial
matches_incl = remove_fp(matches_incl, 'left', r'too superfic', rm_report=True, preview=False)  # 
matches_incl = remove_fp(matches_incl, 'right', r'too superfic', rm_report=True, preview=False)  #  

## appendix
pat_keep = 'in( \w+)? caecum|caecum, adjacent to|caecum and app|abutting base of appendix|distinct from the appendix|within the caecum|= appendix|away from.{,10}append|circumvent.{,15}append'
pat_keep += 'base of appen|appendix: normal|with appendix|appendix is not|h - appendix|2g append|appendix is identif|appendix: abnorm|extending down|obliter|in the caec'
pat_keep += '|caecum and|length appendix|involve the appe|mucoco|appendix - no sig|perforation in the|into the append|into append|invades the append|extends close|relationship to'
matches_incl = remove_fp(matches_incl, 'right', r'^.{,30}appendi', rm_report=True, preview=False, pat_keep=pat_keep)  #  

m = matches_incl.loc[matches_incl.right.str.lower().str.contains('appen'), ['left', 'target', 'right']]
for i, row in m.iterrows():
    print(i, row.left, row.target.upper(), row.right)


#endregion


# ---- 1. Separate out high-confidence matches with pattern 'site of tumour ...' and 'tumour site ...'
#region

## .... 1.1. Pattern 'site of <tumour>'
s = find_first(matches_incl, 'left_proc', r'site of[^\n\r]{,6}$')
print(s.shape)
s = find_first(s, 'right_proc', r'^\: *((?:[\w\(\[\)\]\-]+ ?)+)')
print(s.shape)
s = clean(s, ['left_proc', 'target_proc', 'right_proc'])

for col in ['left_proc', 'target_proc', 'right_proc']:
    print(s[col].unique())


pd.set_option('display.max_colwidth', 300)
s.loc[s.right_proc == 'appendix', ['left', 'target', 'right']]

## Filter out false positives
s['manual_exclusion'] = 0
s.loc[s.right_proc.str.contains('appendix|scar|appendic|ileum|unknown'), 'manual_exclusion'] = 1
s.manual_exclusion.sum()
row_ex = s.loc[s.manual_exclusion==1].row
s.loc[s.manual_exclusion==1].right_proc.unique()

s = s.loc[~s.row.isin(row_ex)]
print(matches_incl.shape)
matches_incl = matches_incl.loc[~matches_incl.row.isin(row_ex)]
print(matches_incl.shape)

## Get site string and reformat
s['site_string'] = s.right_proc.copy()
s['pat_type'] = 'site of tumour:'

## Store
matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape

matches_conf[['left_proc', 'target_proc', 'right_proc', 'site_string']]


## .... 1.2. Pattern: "<tumour> site ..."
s = find_first(matches_incl, 'right_proc', r'^.{,10}site *\: *(?:[\w\(\[\)\]\-]+ ?){1,3}')
print(s.shape)
s['site_string'] = s.right_proc.str.replace(r'[\W\.\n\r]*site[\W\.\n\r]*', '', regex=True)
s = clean(s, ['left_proc', 'target_proc', 'right_proc'])
s['pat_type'] = 'tumour site:'

s.site_string.unique()
print(s[['left_proc', 'target_proc', 'right_proc']])

## Filter out false matches
s['manual_exclusion'] = 0
s.loc[s.right_proc.str.contains('appendix|scar|appendic|ileum|unknown|abdomen'), 'manual_exclusion'] = 1
s.manual_exclusion.sum()
row_ex = s.loc[s.manual_exclusion==1].row
s.loc[s.manual_exclusion==1].right_proc.unique()

s = s.loc[~s.row.isin(row_ex)]
print(matches_incl.shape)
matches_incl = matches_incl.loc[~matches_incl.row.isin(row_ex)]
print(matches_incl.shape)

s.site_string.unique()

## Store
matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape
matches_conf[['left_proc', 'target_proc', 'right_proc', 'site_string']]


## .... 1.3. Pattern: 'site: ... <tumour>'
s = find_first(matches_incl, 'left_proc', r'\bsite ?:[ \-]*(?:\w+[ \(\)\[\]\-\:]?){1,3}')
s.left_proc.unique()
s.right_proc = s.right_proc.str[:47]
s['site_string'] = s.left_proc


## Exclude some false positives
s['manual_exclusion'] = 0
s.loc[s.left_proc.str.lower().str.contains('appendix|scarring|appendic|ileum|unknown|serosa|small bowel|ileum', regex=True), 'manual_exclusion'] = 1
s.manual_exclusion.sum()
row_ex = s.loc[s.manual_exclusion==1].row
s.loc[s.manual_exclusion==1].left_proc.unique()

s = s.loc[~s.row.isin(row_ex)]
print(matches_incl.shape)
matches_incl = matches_incl.loc[~matches_incl.row.isin(row_ex)]
print(matches_incl.shape)


## examine
s['phrase'] = s.left.str.lower() + s.target.str.upper() + s.right.str.lower().str[:80]
s.phrase = s.phrase.str.replace(r'.*(?=site:)', '')
s.phrase = s.phrase.str.replace('\n|\r', ' <n> ')
s.phrase = s.phrase.str.replace('\d+', ' <d> ')
phrases = s.phrase.sort_values().drop_duplicates()
for i, c in enumerate(phrases):
    print(i, c)

phrases[~phrases.str.lower().str.contains('adenocarcinoma')]


## Dbl check reports that don't say adenocarcinoma,
sub = s.loc[~s.phrase.str.lower().str.contains('adenocarcinoma')]
dfsub = df.loc[df.row.isin(sub.row)]

msub = extract(dfsub, report_col, 'summary.{,150}', flags=re.I|re.DOTALL)
msub.target = msub.target.str.replace(r'\r|\n', ' <n> ')
for i, t in enumerate(msub.target.unique()):
    t = dummy_redact(t)
    print('\n====', i, t, '|')

for i, r in enumerate(dfsub.unsafe_report.drop_duplicates()):
    print(len(r))

    safer_r = dummy_redact(r)
    safer_r = re.sub(r'\r', '\n', safer_r)
    print('\n', i, '======')
    print(safer_r)

s['pat_type'] = 'site:'


## Store
matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## .... 1.4. Pattern 'site of <tumour>' where site does not come immediately
s = find_first(matches_incl, 'left_proc', r'site of[^\n\r]{,6}$')
s = find_first(s, 'right_proc', r'^[^,\r\n]{,50}')
print(s.shape)
s = clean(s, ['left_proc', 'target_proc', 'right_proc'])
s['pat_type'] = 'site of tumour ...'

for col in ['left_proc', 'target_proc', 'right_proc']:
    print(s[col].unique())

## Exclude fp
s['manual_exclusion'] = 0
s['phrase'] = s.left_proc + s.target + s.right_proc
s.loc[s.phrase.str.lower().str.contains('site of this|in progress', regex=True), 'manual_exclusion'] = 1
s.manual_exclusion.sum()
row_ex = s.loc[s.manual_exclusion==1].row
s.loc[s.manual_exclusion==1].phrase.unique()

s = s.loc[~s.row.isin(row_ex)]
print(matches_incl.shape)
matches_incl = matches_incl.loc[~matches_incl.row.isin(row_ex)]
print(matches_incl.shape)


## Store
matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


#endregion



# ---- 2. Separate out high-confidence matches for tumour keywords with pattern 'biopsy ...' ('e.g. colon biopsy: adenocarcinoma')
#region

## 2.1. biopsy ... tumour

s = find_first(matches_incl, 'left_proc', pat='[^\n\r]{,40}bio[ps][sp][yi][^\n\r\.]{,100}$')   #(?:[a-z]+[^\n\r]){,4}
s.shape
s.left_proc = s.left_proc.str.replace(r'\b\d+\.*\s*', '', regex=True)
s.left_proc = s.left_proc.str.replace('^\W+', '', regex=True)
s.left_proc.sort_values().unique()
s = find_first(s, 'right_proc', r'^[\s\n\r\.\d]*((?:[a-z]+[ \,]*){,10})')
s.right_proc.sort_values().unique()
s.right_proc = s.right_proc.str.replace(r'\bdr (\w+ ){,3}', '', regex=True)
s = clean(s, ['left_proc', 'target_proc', 'right_proc'])

for p in s.left_proc.sort_values().unique():
    print(p)


## Get site string and reformat
s['site_string'] = s.left_proc + ' ' + s.target_proc.str.upper() + ' ' + s.right_proc
s['site_string_length'] = s.site_string.apply(len)

check = s.sort_values(by=['site_string_length']).site_string.drop_duplicates()
print(check.shape)
for i, c in enumerate(check):
    print(i, c)


## Explore unwanted sites
extract(s, 'site_string', ex_site, flags=re.I|re.DOTALL)
extract(s, 'site_string', r'\b(?:anal|anus)\b', flags=re.I|re.DOTALL)
extract(s, 'site_string', r'biosp|bioss|biopp', flags=re.I|re.DOTALL)


## Filter out false matches (post crt?)
s['manual_exclusion'] = 0

pats = r'\b(?:anal|anus)\b[^\n\r]{,20}biops[^\n\r]{,30}carcinom'
s.loc[s.site_string.str.contains(pats, flags=re.DOTALL|re.I), 'manual_exclusion'] = 1
s.manual_exclusion.sum()
s.loc[s.manual_exclusion==1, 'site_string'].unique()

pats = 'too superficial|performed on biopsy|was taken from the rectum|multiple biopsies taken\.?|stomach tumour|ileum.{,5}biopsy.{,5}well different'
pats += '|external biopsy|cervical|vaginal|anal margin|anal biops|anus.{,7}biops|testinal origin and I note|tumour is planned|chromogranin'
s.loc[s.site_string.str.contains(pats, flags=re.DOTALL), 'manual_exclusion'] = 1
s.manual_exclusion.sum()
s.loc[s.manual_exclusion==1, 'site_string'].unique()

row_ex = s.loc[s.manual_exclusion==1].row
print(s.shape)
s = s.loc[s.manual_exclusion == 0]
print(s.shape)


## Correct site string - use only left part - o.w. can mess
for p in s.left_proc.sort_values().unique():
    print(p)

s['site_string'] = s.left_proc
mask = s.left_proc.str.contains('has been undertak|diagnostic biopsy for')
s.loc[mask, 'site_string'] = s.loc[mask, 'right_proc']

s['pat_type'] = 'biopsy...tumour'

## Store
matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape

matches_conf[['left_proc', 'target_proc', 'right_proc', 'site_string']]


#endregion



# ---- 3. Remove some additional false positives ----
#region
## No macroscopic evid - still microscopic?
mask = matches_conf.right.str.lower().str.contains('no macroscopic evid|no residual|no primary')
print(mask.sum())

matches_conf['manual_exclusion'] = 0
matches_conf.loc[mask, 'manual_exclusion'] = 1

submatches = matches_conf.loc[matches_conf.manual_exclusion==1]
submatches.right.unique()
submatches.left

row_ex = matches_conf.loc[matches_conf.manual_exclusion==1].row
print(matches_conf.shape)
matches_conf = matches_conf.loc[~matches_conf.row.isin(row_ex)]
print(matches_conf.shape)

print(matches_incl.shape)
matches_incl = matches_incl.loc[~matches_incl.row.isin(row_ex)]
print(matches_incl.shape)
#endregion



# ---- 4. Check: is there any site of tumour stated in excluded matches?
#region
print(matches_conf.shape, matches_incl.shape)


## .... 4.1. Pattern 'site of' on the left side -- most are not CRC sites
s2 = find_first(matches_excl, 'left_proc', 'site of[^\n\r]{,6}$')
s2.shape
s2 = find_first(s2, 'right_proc', r'^\: *((?:[\w\(\[\)\]]+ ?)+)|')
s2 = clean(s2, ['left_proc', 'target_proc', 'right_proc'])
s2.shape
s2.right_proc.unique()  ## Can see that unwanted sites there

s2 = s2.loc[~s2.exclusion_reason.str.contains('no site')]  ## ALl excluded due to unwanted sites
print(s2.shape)

## One was not excluded as site was rect, but keywords for irrelevant sites were near. Retain it
s2.iloc[0]
s2['site_string'] = s2.right_proc.copy()
s2['pat_type'] = 'site of tumour:'
matches_conf = pd.concat(objs=[matches_conf, s2], axis=0)


## .... 4.2. Pattern 'site' in right side -- none are CRC sites
s2 = find_first(matches_excl, 'right_proc', '^.{,10}site *\: *(?:[\w\(\[\)\]\-]+ ?){1,5}')
s2.shape
s2['site_string'] = s2.right_proc.str.replace(r'.*site[\W\.\n\r]*', '', regex=True)
s2 = clean(s2, ['left_proc', 'target_proc', 'right_proc'])
s2.right_proc.unique()
s2.site_string.unique()

s2.loc[s2.site_string == 'central '].iloc[0]

#endregion



# ---- 5. Separate out high-confidence matches for tumour keywords with pattern 'summary' (e.g. 'summary: colorectal adenocarcinoma')
#region

s = find_first(matches_incl, 'left_proc', pat='summar[yi].{,100}$')   #(?:[a-z]+[^\n\r]){,4}
s.shape

s.left_proc = s.left_proc.str.replace(r'\b\d+\.*\s*', '', regex=True)
s.left_proc = s.left_proc.str.replace('^\W+', '', regex=True)
s.left_proc.sort_values().unique()

s = find_first(s, 'right_proc', r'^[\s\n\r\.\d]*((?:[a-z]+[ \,]*){,10})')
s.shape
s.right_proc.sort_values().unique()
s.right_proc = s.right_proc.str.replace(r'\bdr (\w+ ){,3}', '', regex=True)
s = clean(s, ['left_proc', 'target_proc', 'right_proc'])

## Get site string and reformat
s['site_string'] = s.left_proc.copy()

check = s.site_string.sort_values().unique()
print(check.shape)
for i, c in enumerate(check):
    print(i, c)


## Filter out some false positive matches (post crt?)
s['manual_exclusion'] = 0
pats = 'vaginal|peritoneal|bronchial|residual|anal|anus'
s.loc[s.site_string.str.contains(pats), 'manual_exclusion'] = 1
s.manual_exclusion.sum()
s.loc[s.manual_exclusion==1, 'site_string'].unique()

row_ex = s.loc[s.manual_exclusion==1].row
print(s.shape)
s = s.loc[s.manual_exclusion == 0]
print(s.shape)

print(matches_incl.shape)
matches_incl = matches_incl.loc[~matches_incl.row.isin(row_ex)]
print(matches_incl.shape)


# Fix site string
mask = s.site_string.str.lower().str.contains('ascending[ \w]{,20}colon: within normal limits')
s.site_string[mask]
s.loc[mask, 'site_string'] = s.loc[mask, 'site_string'].str.replace('ascending[ \w]{,20}colon: within normal limits', '<other norm sites>', regex=True)


## Store
s['pat_type'] = 'summary...'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape

matches_conf[['left_proc', 'target_proc', 'right_proc', 'site_string']]


#endregion



# ---- 6. Separate out high-confidence matches for tumour keywords with nearby anatomical site keyword
#region

s = matches_incl.copy()
s = find_first(s, 'left_proc', r'(?:[\w\(\[\)\]\,]+ ?){,5}[ \.\(\)\[\]\:\-\,]*$')
s.left_proc.unique()

s_sites = extract(s, 'left_proc', pats_site, flags=re.I|re.DOTALL)
s = s.iloc[s_sites.row.drop_duplicates()]
print(s.shape)

s.left_proc = s.left_proc.str.replace(r'\b\d+\.*\s*', '', regex=True)
s.left_proc = s.left_proc.str.replace('^\W+', '', regex=True)
s.left_proc.sort_values().unique()

s = find_first(s, 'right_proc', r'^[\s\n\r\.\d]*((?:[a-z]+[ \,]*){,10})')
s.right_proc.sort_values().unique()
s.right_proc = s.right_proc.str.replace(r'\bdr (\w+ ){,3}', '', regex=True)
s = clean(s, ['left_proc', 'target_proc', 'right_proc'])

## Get site string and reformat
s['site_string'] = s.left_proc.copy()
s['phrase_proc'] = s.left_proc + s.target_proc.str.upper() + s.right_proc

check = s.phrase_proc.sort_values().unique()
print(check.shape)
for i, c in enumerate(check):
    print(i, c)

## Filter out some false positive matches (post crt?)
s['manual_exclusion'] = 0
pats = 'vaginal|peritoneal|bronchial|unremarkable|prostatic|small bowel|has been compared|had a rectal|years ago|recurrence|somewhat unusual'
pats += '|cannot be stated|metasta|recur|could occur in|differs from the caecal|original colonic|cdx2 and ck20 negative colonic adenocarcinomas|screening'
pats += '|metastasis of|anorectal|bowel wall stenosed|tumours and synchronous|concurrent colon|versus a very low|through the glandular|in sporadic colo'
s.loc[s.phrase_proc.str.lower().str.contains(pats), 'manual_exclusion'] = 1
s.manual_exclusion.sum()
s['phrase'] = s.left.str.lower() + s.target.str.upper() + s.right.str.lower()
s.phrase = s.phrase.str.replace('\n|\r', ' <n> ')
s.phrase = s.phrase.str.replace('\d+', ' <d> ')
phrases = s.loc[s.manual_exclusion==1, 'phrase'].unique()
for i, c in enumerate(phrases):
    print('\n')
    print(i, c)

test = extract(df, report_col, 'hemicolectomy specimen contains a submucosal.{,100}intuss', flags=re.I|re.DOTALL, pad_left=200, pad_right=200)
test['phrase'] = test.left + test.target + test.right
test.phrase.values
df.iloc[test.row[0]][report_col]

row_ex = s.loc[s.manual_exclusion==1].row
print(s.shape)
s = s.loc[s.manual_exclusion == 0]
print(s.shape)

print(matches_incl.shape)
matches_incl = matches_incl.loc[~matches_incl.row.isin(row_ex)]
print(matches_incl.shape)

## Store
s['pat_type'] = '<site>...tumour'
matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape

matches_conf[['left_proc', 'target_proc', 'right_proc', 'site_string']]


#endregion



# ---- 7. Remove and keep semi manually
#region

## 7.1. Remove small bowel, other matches on right
matches_incl = remove_fp(matches_incl, 'right', r'^[^\n\r]{,20}(?:of|within)[^\n\r]{,20}(?:small bowel|ileum|ileal)', 
                         rm_report=True, preview=False)  ## 
matches_incl = remove_fp(matches_incl, 'right', r'^.{,50}(?:small bowel|ileum|ileal)', rm_report=True, preview=False)  ## 

df.iloc[matches_incl.loc[matches_incl.tmp_exclusion==1].row][report_col].values


## 7.2. Remove small bowel, other matches on left - do not exclude
pat_keep = 'small bowel and.{,20}large bowel|ileum and.{,30}large bowel|loop of small bowel|caecum and|small bowel.{,40} normal'
pat_keep += '|possibly also.{,20}small bowel|nodular area within fat with adjacent small bowel|ileal side of'
matches_incl = remove_fp(matches_incl, 'left', r'(?:small bowel|ileum|ileal).{,50}$', rm_report=True, preview=False, pat_keep=pat_keep)  ## 
 

## 7.3. Remove liver
pat = r'(?:\bliver|\bsegments? (?:[1-8]|i{1,3}|iv|v|vi{1,3})|\bhepatectom|hepatic (?!flex\w*))'
pat_keep = r'with segment|3\. segment'  #  
matches_incl = remove_fp(matches_incl, 'left', pat, rm_report=True, preview=False, pat_keep=pat_keep)

tmp = matches_incl.loc[matches_incl.tmp_exclusion==1]
check = df.iloc[tmp.row][report_col]
for row in check:
    print('\n\n======------=====-----')
    print(row)


## 7.4. Remove benign
matches_incl = remove_fp(matches_incl, 'left', r'benign', rm_report=True, preview=False)  ## 
matches_incl = remove_fp(matches_incl, 'right', r'benign', rm_report=True, preview=False)  ## 


## 7.5. Remove other unwanted sites
out = extract(matches_incl, 'left', ex_site, flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(matches_incl.row)})
df['row'] = np.arange(df.shape[0])
dfsub = df.loc[df.row.isin(out.row)]
for i, row in dfsub.iterrows():
    report = row[report_col]
    report = re.sub(r'\n|\r', ' <n> ', report)
    print('\n\n===---====', i)
    print(report)

out['exclude'] = 1
out.loc[out.row.isin([86503]), 'exclude'] = 0
out_excl = out.loc[out.exclude==1]
matches_incl = matches_incl.loc[~matches_incl.row.isin(out_excl.row)]
print(matches_incl.shape)


out2 = extract(matches_incl, 'right', ex_site, flags=re.I|re.DOTALL)
out2['row'] = out2.row.replace({i:row for i, row in enumerate(matches_incl.row)})

dfsub = df.loc[df.row.isin(out2.row)]
for i, row in dfsub.iterrows():
    report = row[report_col]
    report = re.sub(r'\n|\r', ' <n> ', report)
    print('\n\n===---====', i)
    print(report)

out2['exclude'] = 1
out2_excl = out2.loc[out.exclude==1]
matches_incl = matches_incl.loc[~matches_incl.row.isin(out2_excl.row)]
print(matches_incl.shape)


## 7.6. Exclude additional small bowel reports
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'summar.{,30}small bowel', flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
matches_incl = matches_incl.loc[~matches_incl.row.isin(out.row)]
matches_incl.shape[0]

dfsub.loc[dfsub.row.isin(out.row), report_col].iloc[0]


## 7.7. Remove some reports with site in bowel wall - looks like these are not CRC reports
out = extract(matches_incl, 'left', 'bowel wall', flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(matches_incl.row)})

dfsub = df.loc[df.row.isin(out.row)]
for i, row in dfsub.iterrows():
    report = row[report_col]
    report = re.sub(r'\n|\r', ' <n> ', report)
    print('\n\n===---====', i)
    print(report)

matches_incl = matches_incl.loc[~matches_incl.row.isin(out.row)]
matches_incl.shape[0]


## 7.8. Remove some reports with site in bowel wall - looks like these are not CRC reports
out = extract(matches_incl, 'right', 'bowel wall', flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(matches_incl.row)})

dfsub = df.loc[df.row.isin(out.row)]
for i, row in dfsub.iterrows():
    report = row[report_col]
    report = re.sub(r'\n|\r', ' <n> ', report)
    print('\n\n===---====', i)
    print(report)

matches_incl = matches_incl.loc[~matches_incl.row.isin(out.row)]
matches_incl.shape[0]


## 7.9. Explore site - some high confidence matches here
out = extract(matches_incl, 'right', 'site ?\: ?[\w ]*', flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(matches_incl.row)})

dfsub = df.loc[df.row.isin(out.row)]
for i, row in dfsub.iterrows():
    report = row[report_col]
    report = re.sub(r'\n|\r', ' <n> ', report)
    print('\n\n===---====', i)
    print(report)

out2 = extract(dfsub, report_col, 'type ?\: *\w*carcinom\w*', flags=re.I|re.DOTALL)
out2['row'] = out2.row.replace({i:row for i, row in enumerate(dfsub.row)})

s = out2.copy()
s = s[['row', 'start', 'end', 'left', 'target', 'right']]
s = s.merge(out[['row', 'target']].rename(columns={'target': 'site_string'}))
s['pat_type'] = 'site:'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## 7.10 Exclude additional false positives (FP) that are not CRC, e.g. dysplasia, liver, adenoma, left upper lobe etc
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'summar.{,150}', flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})

pat = 'egfr|anterior resect|transverse colo|sigmoid nodul|lymphoma|colon.{,5}biops.{,20}carc|tem.{,10}carc|rect.{,5}biops.{,10}carc|tem.{,50}carcin|emr'
out['exclude'] = 1
out.loc[out.target.str.lower().str.contains(pat, flags=re.I|re.DOTALL), 'exclude'] = 0
out.loc[out.exclude==0, ['row', 'target']]
out.loc[out.exclude==1, ['row', 'target']]
out[['row', 'target']]

dfsub.loc[dfsub.row == 26694, report_col].values

out_excl = out.loc[out.exclude == 1]
matches_incl = matches_incl.loc[~matches_incl.row.isin(out_excl.row)]
matches_incl.shape[0]


## 7.11. Remove recurrence
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'recurrence|recurrent', flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out[['row', 'target']]

matches_incl = matches_incl.loc[~matches_incl.row.isin(out.row)]
matches_incl.shape[0]


## 7.12. Few extra high confidence matches: where site keyword near cancer keyword
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'summar.{,150}((?:colon|sigmo|right hemi|tem|tamis|anterior resect)\w*).{,60}(\w*carcinom\w*).{,100}', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})

out[['row', 'target', 'target_group1', 'target_group2']]
for t in out.target:
    print(re.sub(r'\n|\r', ' <n> ', t))

s = out.copy()
s['site_string'] = s.target_group1.copy().str.lower()
s['target'] = s.target_group2.copy().str.lower()
s = s[['row', 'target', 'site_string']]
s['pat_type'] = 'summary...'
matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## 7.13. More high confidence matches: where site and tumour keyword are farther apart
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, '(site ?\: ?.{,50}).{,500}(\w*(?:carcinom|tumour)\w*)', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out[['row', 'target', 'target_group1', 'target_group2']]
for i, row in out.iterrows():
    print(row.target)

s = out.copy()
s['site_string'] = s.target_group1.copy()
s['target'] = s.target_group2
s = s[['row', 'target', 'site_string']]
s['pat_type'] = 'site:'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## 7.14. Immunohistochemistry in keeping with colorectal - keep
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, '(\w*tumour|\w*carcinom).{,1000}in keeping.{,20}(colorect\w*|colon\w*).{,20}origin.{,20}', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out[['target']]

s = out.copy()
s['site_string'] = np.nan
s['target'] = s.target_group1
s = s[['row', 'target', 'site_string']]
s['pat_type'] = 'express in keeping with colorectal origin'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## 7.15 Keep codon, egfr
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, '(\w*(?:tumour|carcinom)\w*).{,1000}summar.{,500}(?:codon|egfr)', flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out[['row', 'target']]
out.target_group1

reports = dfsub.loc[dfsub.row.isin(out.row)][report_col]
for r in reports:
    print('\n======')
    print(re.sub('\r|\n', ' <n> ', r))

s = out.copy()
s['site_string'] = np.nan
s['target'] = out.target_group1
s = s[['row', 'target', 'site_string']]
s['pat_type'] = 'codon or egfr'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## 7.16. Anterior researction - keep
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, '(anterior res\w*).{,100}(\w*(?:tumour|carcinom)\w*)', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out.target
out.shape
out = out.loc[~out.target.str.contains('no tumour')]
out.shape

s = out.copy()
s['site_string'] = s.target_group1.str.lower()
s['target'] = s.target_group2.str.lower()
s = s[['row', 'target', 'site_string']]
s['pat_type'] = 'anterior resection'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## 7.17. Remove reports that discuss spread from colorectal tumour
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'spread from.{,50}colorect\w*.{,20}', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out.target

matches_incl = matches_incl.loc[~matches_incl.row.isin(out.row)]
matches_incl.shape[0]


## 7.18. More immunohistochemistry reports - keep
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'keeping with.{,50}colorect\w*.{,20}', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out.target
out.shape
out = out.loc[~out.target.str.contains('no tumour')]
out.shape

s = out.copy()
s['site_string'] = s.target_group1.str.lower()
s['target'] = s.target_group2.str.lower()
s = s[['row', 'target', 'site_string']]
s['pat_type'] = 'in keeping with colorectal carcinoma'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## 7.19. Remove gynaecological
dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'gynaecol', flags=re.I|re.DOTALL)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out[['row', 'left', 'target']]

matches_incl = matches_incl.loc[~matches_incl.row.isin(out.row)]
matches_incl.shape[0]


## 7.20. rectosigmoid keep
out = extract(matches_incl, 'left', '(rectosig\w*)[^\n\r]{,50}$', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(matches_incl.row)})
out.target

s = out.copy()
s['site_string'] = s.target.str.lower()
s = s[['row', 'site_string']]
s = s.merge(matches_incl.loc[matches_incl.row.isin(s.row), ['row', 'target']])
s['pat_type'] = 'rectosigmoid ... tumour'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_incl.shape
matches_incl = matches_incl.loc[~matches_incl.row.isin(matches_conf.row)]
matches_incl.shape


## 7.21. Remove 'in people with', and colon in normal limits
matches_incl = remove_fp(matches_incl, 'left', r'people with', rm_report=True, preview=False)  ## 
matches_incl = remove_fp(matches_incl, 'right', r'colon.{,10}within normal', rm_report=True, preview=False)  ## 

matches_incl = remove_fp(matches_incl, 'right', r'^.{,10}reporting template', rm_report=False, preview=False)  ## 
matches_incl = remove_fp(matches_incl, 'right', r'^.{,100}unremarkable', rm_report=False, preview=False)  ## 
matches_incl = remove_fp(matches_incl, 'left', r'from another site', rm_report=False, preview=False)  ## 
matches_incl = remove_fp(matches_incl, 'left', r'myxoid', rm_report=True, preview=False)  ##


## Look at remaining matches --- they all seem ok, even though not all primary reports
char = 100
s = matches_incl.copy()
s.left = s.left.str[-char:]
s.right = s.right.str[:char]
s['phrase'] = s.left.str.lower() + s.target.str.upper() + s.right.str.lower()
s.phrase = s.phrase.str.replace('\n|\r', ' <n> ')
for i, p in enumerate(s.phrase.sort_values().drop_duplicates()):
    print(i, p)


dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'myxoid', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out.target

dfsub.loc[dfsub.row == 3312, report_col].values

dfsub = df.loc[df.row.isin(matches_incl.row)]
out = extract(dfsub, report_col, 'immunopositivity for panck', flags=re.I|re.DOTALL, groups=True)
out['row'] = out.row.replace({i:row for i, row in enumerate(dfsub.row)})
out.target


s = matches_incl.copy()
s['site_string'] = np.nan
s['pat_type'] = '<other>'

matches_conf = pd.concat(objs=[matches_conf, s], axis=0)
matches_conf.shape
del matches_incl

#endregion



# ---- 8. Get site from site string ----
#region

# Clean site strings with potential multiple sites
mask = matches_conf.site_string.fillna('').str.lower().str.contains('ascending colon sessile serrated polyp')
matches_conf.loc[mask, 'site_string'] = matches_conf.loc[mask, 'site_string'].str.replace('ascending colon sessile serrated polyp', '<norm site> sessile serrated polyp')

site_matches = pd.DataFrame()
for site, pat in pats_site_dict.items():
    m = extract(matches_conf, 'site_string', pat, flags=re.I|re.DOTALL)
    m['site'] = site
    site_matches = pd.concat(objs=[site_matches, m], axis=0)

m = extract(matches_conf, 'site_string', r'\btem\b', flags=re.I|re.DOTALL)
m['site'] = 'rectum'
site_matches = pd.concat(objs=[site_matches, m], axis=0)

m = extract(matches_conf, 'site_string', r'ileocaec', flags=re.I|re.DOTALL)
m['site'] = 'ileocaecal'
site_matches = pd.concat(objs=[site_matches, m], axis=0)

m = extract(matches_conf, 'site_string', r'cacecum', flags=re.I|re.DOTALL)
m['site'] = 'caecum'
site_matches = pd.concat(objs=[site_matches, m], axis=0)

sites = site_matches.groupby('row').site.apply(list).reset_index()
sites = sites.rename(columns={'row': 'site_row'})
matches_conf['site_row'] = np.arange(matches_conf.shape[0])

# Check matches w both colon and rectum
mask = sites.site.apply(lambda x: 'colon' in x and 'rectum' in x)
matches_conf.loc[matches_conf.site_row.isin(sites.loc[mask].site_row)].pat_type.value_counts()
matches_conf.loc[matches_conf.site_row.isin(sites.loc[mask].site_row)].site_string.drop_duplicates().unique()

# Add site to matches_conf
matches_conf = matches_conf.merge(sites, how='left', on='site_row')
matches_conf[['site_string', 'site']]
matches_conf = matches_conf.drop(labels=['site_row'], axis=1)

# Explore site strings with no matches
matches_conf.loc[matches_conf.site.isna()].site_string.unique()

# Get site at report level
sites_report = matches_conf[['row', 'site']].dropna(subset=['site'])
sites_report = sites_report.groupby('row').site.apply(lambda x: sum(x, [])).reset_index()
sites_report.site = sites_report.site.apply(lambda x: ';'.join(list(set(x))))
sites_report = sites_report.rename(columns={'site': 'site_nlp'})
sites_report.row.nunique() == sites_report.shape[0]

# Simplify site 
concepts = pats_site_dict.keys()
concept_map = {'caecum': 'colon', 
               'right (ascending) colon': 'colon',
               'hepatic flexure': 'colon',
               'transverse colon': 'colon', 
               'splenic flexure': 'colon', 
               'left (descending) colon': 'colon',
               'sigmoid colon': 'colon', 
               'colon': 'colon', 
               'unknown': 'unknown',
               'rectosigmoid': 'rectosigmoid',
               'rectum': 'rectum',
               'ileocaecal': 'colon'
               }
[k in concept_map.keys() for k in pats_site_dict.keys()]
assert all(k in concept_map.keys() for k in pats_site_dict.keys())

sites_report['site_nlp_simple'] = ''
mask0 = sites_report.site_nlp.str.lower().str.contains('colon|flexure|caecum|ileocaec')
mask1 = sites_report.site_nlp.str.lower().str.contains('rectum')
mask2 = sites_report.site_nlp.str.lower().str.contains('rectosigmoid')
sites_report.loc[mask0 & ~mask1, 'site_nlp_simple'] = 'colon'
sites_report.loc[mask1 & ~mask0, 'site_nlp_simple'] = 'rectum'
sites_report.loc[mask1 & mask0, 'site_nlp_simple'] = 'colon, rectum'
sites_report.loc[~mask0 & ~mask1 & mask2, 'site_nlp_simple'] = 'rectosigmoid'
sites_report.site_nlp_simple = sites_report.site_nlp_simple.replace({'': np.nan})
sites_report.site_nlp_simple.value_counts()

sites_report.loc[sites_report.site_nlp_simple.isna(), 'site_nlp'].unique()

# Check where no simple iste assigned
test = sites_report.loc[sites_report.site_nlp_simple.isna(), 'row'].unique()
matches_conf.loc[matches_conf.row.isin(test), ['row', 'left', 'target', 'right']]

#endregion



# ---- 9. Explore what is left (not relevant - as all remaining matches absorbed)
#region

explore = False
if explore:
    s = find_first(matches_incl, 'left_proc', '(?:\w+[ ,-]){1,4}$')
    tmp = s.left_proc.str.replace('\r|\n', ' <n> ').sort_values()#.unique()
    tmp.value_counts()

    s = find_first(matches_incl, 'right_proc', '^\W*(?:\w+[ ,-]){1,6}')
    tmp = s.right_proc.str.replace('\r|\n', ' <n> ').sort_values()#.unique()
    tmp.value_counts()

    out = matches_incl.left.str.lower().str.replace('\r|\n', ' <n> ').str[-60:].sort_values()
    for i, c in enumerate(out):
        print(i, c)

    out = matches_incl.right.str.lower().str.replace('\r|\n', ' <n> ').str[:60].sort_values()
    for i, c in enumerate(out):
        print(i, c)

    matches_incl.loc[matches_incl.left.str.lower().str.contains('spread from')].left.unique()
#endregion



# ---- 10. Process: reorder columns, separate manual exclusions, add reports to matches, add CRC and site to reports
#region

# Reorder columns
matches_conf['high_conf'] = 1
matches_conf = matches_conf[['idx', 'row', 'start', 'end', 'left', 'target', 'right', 'exclusion_indicator', 'exclusion_reason',
                           'subrow', 'left_proc', 'target_proc', 'right_proc', 'site_string', 'phrase', 'site', 'high_conf', 'pat_type']]

# Get manually excluded matches (but only those where nothing was included for a report)
mask = ~matches.index.isin(matches_excl.index) & ~matches.row.isin(matches_conf.row)
matches_excl_manual = matches.loc[mask]
matches_excl_manual.shape
matches_excl_manual = clean(matches_excl_manual, ['left', 'target', 'right'])

# Add reports to matches
df['row'] = np.arange(df.shape[0])
matches_conf = matches_conf.merge(df[['row', report_col]], how='left')
matches_conf[report_col] = matches_conf[report_col].str.lower()
matches_conf[report_col] = matches_conf[report_col].str.replace('\d+', '<d>')
matches_conf[report_col] = matches_conf[report_col].str.replace(r'\n+|\r+', ' <n> ')

save_matches = True
if save_matches:
    matches.to_csv(out_path / 'included_matches_crc_20240618.csv', index=False)
    matches_conf.to_csv(out_path / 'manually_included_matches_crc_20240618.csv', index=False)
    matches_excl_manual.to_csv(out_path / 'manually_excluded_matches_crc_20240618.csv', index=False)
    matches_excl.to_csv(out_path / 'excluded_matches_crc_20240618.csv', index=False)

# Add CRC status to reports 
df['row'] = np.arange(df.shape[0])
df['crc_nlp'] = 0
df.loc[df.row.isin(matches_conf.row), 'crc_nlp'] = 1
df.crc_nlp.sum()
df.crc_nlp.mean()
ncrc = df.loc[df.crc_nlp==1, 'mrn_number'].nunique()
df[['crc_nlp', 'crc_nlp_orig']].value_counts()

# Add site to reports
sites_report_pos = sites_report.loc[sites_report.row.isin(matches_conf.row)]

df.shape
df = df.merge(sites_report_pos, how='left')
df.shape

mask = df.crc_nlp == 1
df.loc[mask, ['site_nlp', 'site_nlp_simple']] = df.loc[mask, ['site_nlp', 'site_nlp_simple']].fillna('Not extracted')
df.site_nlp.value_counts()
df.site_nlp_simple.value_counts()

#endregion



# ---- 11. Get data from snomed codes
#region

# SNOMED T and M codes
# From Loughrey, Quirke, Shepherd (2018), 
# "Standards and datasets for reporting cancers: Dataset for histopathological reporting of colorectal cancer"
# https://www.rcpath.org/uploads/assets/c8b61ba0-ae3f-43f1-85ffd3ab9f17cfe6/G049-Dataset-for-histopathological-reporting-of-colorectal-cancer.pdf
# Creating objects here, as putting inside function may add tab to each line which can interfere with regex
txt_t = """
Colon T59300 (SNOMED 3)
T67000 (SNOMED 2)
Colon structure
(body structure)
71854001
Caecum T59100 (SNOMED 3)
T67100 (SNOMED 2)
Cecum structure
(body structure)
32713005
Ascending colon T59420(SNOMED 3)
T67200 (SNOMED 2)
Ascending colon structure
(body structure)
9040008
Hepatic flexure T59438 (SNOMED 3)
T67300 (SNOMED 2)
Structure of right colic flexure
(body structure)
48338005
Transverse colon T59440 (SNOMED 3)
T67400 (SNOMED 2)
Transverse colon structure
(body structure)
485005
Splenic flexure T59442 (SNOMED 3)
T67500 (SNOMED 2)
Structure of left colic flexure
(body structure)
72592005
Descending colon T59460 (SNOMED 3)
T67600 (SNOMED 2)
Descending colon structure
(body structure)
32622004
Sigmoid colon T59470 (SNOMED 3)
T67700 (SNOMED 2)
Sigmoid colon structure
(body structure)
60184004
Rectosigmoid T59680 (SNOMED 3)
T68200 (SNOMED 2)
Rectosigmoid structure
(body structure)
81922002
Rectum T59600 (SNOMED 3)
T68000 (SNOMED 2)
Rectum structure
(body structure)
34402009
"""

txt_m = """
Adenoma M81400 Adenoma, no subtype
(morphologic abnormality)
32048006
Dysplasia M74000 Dysplasia (morphologic
abnormality)
25723000
Dysplasia, high grade M74003 Severe dysplasia (morphologic
abnormality)
28558000
Carcinoma M80103 Carcinoma, no subtype
(morphologic abnormality)
68453008
Adenocarcinoma M81403 Adenocarcinoma, no subtype
(morphologic abnormality)
35917007
Mucinous
adenocarcinoma
M84803 Mucinous adenocarcinoma
(morphologic abnormality)
72495009
Signet ring cell
adenocarcinoma
M84903 Signet ring cell carcinoma
(morphologic abnormality)
87737001
Adenosquamous
carcinoma
M85603 Adenosquamous carcinoma
(morphologic abnormality)
59367005
Squamous cell
carcinoma
M80703 Squamous cell carcinoma, no
ICD-O subtype (morphologic
abnormality)
28899001
Undifferentiated
carcinoma
M80203 Carcinoma, undifferentiated
(morphologic abnormality)
38549000
Goblet cell carcinoid M82433 Goblet cell carcinoid
(morphologic abnormality)
31396002
Mixed carcinoidadenocarcinoma
M82443 Composite carcinoid
(morphologic abnormality)
51465000
Micropapillary carcinoma M82653 Micropapillary carcinoma
(morphologic abnormality)
450895005
Serrated
adenocarcinoma
M82133 Serrated adenocarcinoma
(morphologic abnormality)
450948005
Spindle cell carcinoma M80323 Spindle cell carcinoma
(morphologic abnormality)
65692009
Medullary carcinoma M85103 Medullary carcinoma
(morphologic abnormality)
32913002
Cribriform comedo-type
adenocarcinoma
M82013 Cribriform carcinoma
(morphologic abnormality)
30156004
"""


# T codes
tmp = pd.DataFrame([txt_t], columns=['txt'])
pat = r'([a-zA-Z]+(?:[ \-\,\n]{1,3}[a-zA-Z]+){,3})\W{,3}T(\d+)\W{,3}SNOMED \d\W{,3}T(\d+)\W{,3}SNOMED \d.{,}?\W(\d+)'
tmp = extract(tmp, 'txt', pat, flags=re.IGNORECASE|re.DOTALL)
tmp = tmp.rename(columns={'target_group1':'site', 
                        'target_group2':'snomed2_t', 
                        'target_group3':'snomed3_t', 
                        'target_group4':'snomed_ct'})
tmp.site = tmp.site.str.lower()
tmp.snomed2_t = tmp.snomed2_t.str.upper()
tmp.snomed3_t = tmp.snomed3_t.str.upper()
snomedt = tmp[['site', 'snomed2_t', 'snomed3_t', 'snomed_ct']]
print(snomedt)

# M codes
tmp = pd.DataFrame([txt_m], columns=['txt'])
tmp = extract(tmp, 'txt', r'([a-zA-Z]+(?:[ \-\,\n]{1,3}[a-zA-Z]+){,3})\W{,3}M(\d+).*?\W(\d+)', flags=re.IGNORECASE|re.DOTALL)
tmp = tmp.rename(columns={'target_group1':'abnormality', 'target_group2':'snomed_m', 'target_group3':'snomed_ct'})
tmp.abnormality = tmp.abnormality.str.lower().str.replace('\n', ' ')
tmp.snomed_m = tmp.snomed_m.str.upper()
snomedm = tmp[['abnormality', 'snomed_m', 'snomed_ct']]
print(snomedm)

# Get snomed reference
df['snomed_t_orig'] = df.snomed_t.copy()
df['snomed_m_orig'] = df.snomed_m.copy()

st = snomedt.snomed2_t.to_list() + snomedt.snomed3_t.to_list() + snomedt.snomed_ct.to_list()  # Get all T codes
sm = snomedm.snomed_m.to_list() + snomedm.snomed_ct.to_list()  # Get all M codes

print('----REPORTED SNOMED CODES----')
print('\nUnclean SNOMED T codes: {}'.format(df.snomed_t.unique()))
df.snomed_t = df.snomed_t.str.replace('\W', '', regex=True)
df.snomed_t = df.snomed_t.str.replace('[a-zA-Z]{5,}', '', regex=True)
print('Cleaned SNOMED T codes: {}'.format(df.snomed_t.unique()))
print('Top 10 value counts for cleaned SNOMED T codes: \n{}'.format(df.snomed_t.value_counts()[0:10]))
df.snomed_t = df.snomed_t.str.replace('^67$', '67000', regex=True)  # There was one case with code 67 - assume 67000

print('\nUnclean SNOMED M codes: {}'.format(df.snomed_m.unique()))
df.snomed_m = df.snomed_m.str.replace('\W', '', regex=True)
df.snomed_m = df.snomed_m.str.replace('[a-zA-Z]{5,}', '', regex=True)
print('Cleaned SNOMED M codes: \n{}'.format(df.snomed_m.unique()))
print('Top 10 value counts for cleaned SNOMED M codes: \n{}'.format(df.snomed_m.value_counts()[0:10]))

pd.set_option('display.min_rows', 100, 'display.max_rows', 100)
df.snomed_m.value_counts()
df.snomed_m.isin(sm).sum()

# Code to meaning (NB. only for CRC-relevant codes!)
tmp = snomedt.melt(id_vars='site', value_name='snomed_code')[['site', 'snomed_code']]
repl_site = {c: s for c, s in zip(tmp.snomed_code, tmp.site)}
tmp = snomedm.melt(id_vars='abnormality', value_name='snomed_code')[['abnormality', 'snomed_code']]
repl_morph = {c: s for c, s in zip(tmp.snomed_code, tmp.abnormality)}

df['site_snomed'] = np.nan
mask = df.snomed_t.isin(st)
df.loc[mask, 'site_snomed'] = df.loc[mask, 'snomed_t'].replace(repl_site)
df.site_snomed.value_counts()
df.site_snomed = df.site_snomed.replace({'': np.nan})

df['site_snomed_simple'] = df.site_snomed.replace({'sigmoid colon': 'colon', 'ascending colon': 'colon',
                                                'transverse colon': 'colon', 'caecum': 'colon', 'hepatic flexure': 'colon',
                                                'descending colon': 'colon', 'splenic flexure': 'colon'})
df.site_snomed_simple.value_counts()

df['abnormality_snomed'] = np.nan
mask = df.snomed_m.isin(sm) & ~df.site_snomed.isna()
df.loc[mask, 'abnormality_snomed'] = df.loc[mask, 'snomed_m'].replace(repl_morph)


## Compare SNOMED and NLP sites
dfsub = df.loc[df.crc_nlp==1, ['row', 'site_nlp', 'site_nlp_simple', 'site_snomed', 'site_snomed_simple', report_col]]
dfsub.isna().sum()
dfsub = dfsub.fillna('')
test = dfsub.apply(lambda x: x.site_snomed_simple in x.site_nlp_simple, axis=1)
test.mean()
dfsubsub = dfsub.loc[~test]
dfsubsub.site_snomed_simple.value_counts()
dfsubsub[['site_nlp_simple', 'site_snomed_simple']].value_counts()
# Indicate cancer
df['crc_snomed'] = 0
mask = df.abnormality_snomed.fillna('').str.contains('carcinom', regex=True)
df.loc[mask, 'crc_snomed'] = 1
#endregion



# ---- 12. Reformat the tables so that they can be included in the pseudonymised data product
#region
df_air = df.drop(labels=[report_col], axis=1)
df_air.columns
df_air = df_air.loc[(df_air.crc_nlp==1)|(df_air.crc_nlp_orig==1)]
df_air.shape
df_air.columns
df_air.site_snomed.unique()
df_air.site_nlp.unique()
df_air[['crc_nlp_orig', 'crc_nlp']].value_counts()

matches_conf.site.value_counts()
matches_conf.high_conf.value_counts()
matches_conf.target.value_counts()
mask = matches_conf.target.str.lower().str.contains('keeping with', regex=True)
matches_conf.loc[mask, 'target'] = matches_conf.loc[mask, 'target'].str.lower().str.replace('keeping with\W*|\n|\r', '')
matches_conf.loc[mask, 'target'] = matches_conf.loc[mask, 'target'].str.lower().replace('\n|\r', '')
matches_conf.loc[mask, 'target'] = matches_conf.loc[mask, 'target'].str.lower().replace('\[re|\s*in t|\W*there ar$|type\:\W*', '', regex=True)
matches_conf.target = matches_conf.target.str.lower()
matches_conf.target = matches_conf.target.str.replace('type: |rather th\w*|\.poly\w*|\. the tu\w*|dr', '', regex=True)

matches_conf.target.value_counts()

matches_conf.loc[matches_conf.target.str.startswith('primary colorectal tumours, r')].iloc[0]  # relevant match: in keeping w primary!

matches_air = matches_conf.drop(labels=[report_col, 'left', 'right', 'subrow',
                                        'left_proc', 'right_proc', 'target_proc', 'site_string', 'phrase',
                                        'exclusion_indicator', 'exclusion_reason'], axis=1)
matches_air.columns
matches_air.target.unique()
matches_air.site.astype(str).unique()
matches_air.pat_type.value_counts()

## Save
df_air.to_csv(out_path / 'crc_from_path_20240618.csv', index=False)
matches_air.to_csv(out_path / 'crc_matches_20240618.csv', index=False)

## to parquet 
#pip install pyarrow
m = pd.read_csv(out_path / 'crc_matches_20240618.csv')
m.to_parquet(out_path / 'crc_matches_20240618_1.parquet', index=False)

## To airlock dir
save_airlock = False
if save_airlock:
    airlock_path = Path("<path redacted>/airlock-part3_fit_data_update_20240530")
    airlock_path.mkdir(exist_ok=True)
    df_air.to_csv(airlock_path / 'crc_from_path_20240618.csv', index=False)
    matches_air.to_csv(airlock_path / 'crc_matches_20240618.csv', index=False)

#endregion
