"""Extract TNM staging scores from radiology, pathology and endoscopy reports

All matches for TNM staging scores across reports are extracted using the get_tnm_phrase_par function
from textmining.tnm.tnm, in lines 21 to 57.

After line 120, a few false positive matches are removed. 

After line 174, the get_tnm_values function is used to assign the maximum and minimum TNM score for each report,
based on previously extracted matches.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from textmining.tnm.tnm import get_tnm_phrase_par, get_tnm_values


out_path = Path('<path redacted>')
data_path = Path('<path redacted>')

tnm_kwargs = {'remove_unusual': True, 'remove_historical': False, 'remove_falsepos': True}
run = False
nrows = None  # set to 1000 to test that code runs

# TNM phrases from pathology reports
df = pd.read_csv(data_path / 'pathology_unredacted_reports.csv', sep=',', engine='c', lineterminator='\n', nrows=nrows)
print(df.columns)
fname = 'matches_tnm_path_20240619.csv'
if run:
    matches = get_tnm_phrase_par(nchunks=8, njobs=4, df=df, col='unsafe_imaging_report', **tnm_kwargs)
    matches.to_csv(out_path / fname, index=False)
else:
    matches = pd.read_csv(out_path / fname)

# TNM phrases from imaging reports
df_img = pd.read_csv(data_path / 'radiology_unredacted_reports.csv', sep=',', engine='c', lineterminator='\n', nrows=nrows)

##Retain imaging reports relevant for CRC
img_type_path = Path("<path redacted>") 
img_types = pd.read_excel(img_type_path)
img_types = img_types.loc[img_types['needed for NIHR HIC CRC'] != 'no']
img_types['needed for NIHR HIC CRC'].unique()
df_img = df_img.loc[df_img.imaging_code.str.upper().isin(img_types.imaging_code.str.upper())]
df_img.shape[0]

fname = 'matches_tnm_img_20240619.csv'
if run:
    matches_img = get_tnm_phrase_par(nchunks=8, njobs=4, df=df_img, col='imaging_report', **tnm_kwargs)
    matches_img.to_csv(out_path / fname, index=False)
else:
    matches_img = pd.read_csv(out_path / fname)

# TNM phrases from endoscopy reports (57 min)
df_endo = pd.read_csv(data_path / 'endoscopy_unredacted_reports.csv', sep=',', engine='c', lineterminator='\n', nrows=nrows)
print(df_endo.columns, df_endo.shape)
fname = 'matches_tnm_endo_20240619.csv'
if run:
    matches_endo = get_tnm_phrase_par(nchunks=8, njobs=4, df=df_endo, col='report_text', **tnm_kwargs)
    matches_endo.to_csv(out_path / fname, index=False)
else:
    matches_endo = pd.read_csv(out_path / fname)


# Review excluded matches 
for m in [matches, matches_img, matches_endo]:
    print('\n======')

    msub = m.loc[m.exclusion_indicator == 1]
    msub = msub[['left', 'target', 'right', 'exclusion_reason']].drop_duplicates()
    msub.left = msub.left.str.replace('\n|\r', '<n>')
    msub.target = msub.target.str.replace('\n|\r', '<n>')
    msub.right = msub.right.str.replace('\n|\r', '<n>')
    for i, (j, row) in enumerate(msub.iterrows()):
        print('\n', i, row.left.lower(), row.target.upper(), row.right.lower(), '|', row.exclusion_reason)

matches.exclusion_reason.value_counts()


# Review targets
print_ctx=True
chr = 30
for m in [matches, matches_img, matches_endo]:
    print('\n======')
    t = m[['left', 'target', 'right']].drop_duplicates(subset='target')
    t['target_len'] = t.target.fillna('').apply(len)

    t.left = t.left.str.replace('\n|\r', '<n>')
    t.target = t.target.str.replace('\n|\r', '<n>')
    t.right = t.right.str.replace('\n|\r', '<n>')

    t = t.sort_values(by='target_len', ascending=False)
    for i, row in t.iterrows():
        if print_ctx:
            print('\n', i, row.target, '|', row.left[-chr:], '<<<', row.target, '>>>', row.right[:chr])
        else:
            print(i, row.target)


# Dbl check where target is nan -- some coding errors here
for m in [matches, matches_img, matches_endo]:
    print('\n====')
    msub = m.loc[m.target.isna()]
    print(msub.shape)

    t = pd.concat([msub, m.loc[m.row.isin(msub.row)]], axis=0)
    print(t.shape)
    t = t.drop_duplicates().sort_values(by=['row'])

    t.left = t.left.str.replace('\n|\r', '<n>')
    t.target = t.target.str.replace('\n|\r', '<n>')
    t.right = t.right.str.replace('\n|\r', '<n>')

    for i, row in t.iterrows():
        if print_ctx:
            print('\n', i, row.row, '|', row.target, '|', row.left[-chr:], '<<<', row.target, '>>>', row.right[:chr])
        else:
            print(i, row.row, '|', row.target)


# Remove some false pos, such as 'cmx', 'arynx'
#region

## Review lowercase targets - some of these likely false pos
for m in [matches, matches_img, matches_endo]:
    print('\n======')
    t = m[['left', 'target', 'right']].drop_duplicates(subset='target')
    t['target_len'] = t.target.fillna('').apply(len)

    t.left = t.left.str.replace('\n|\r', '<n>')
    t.target = t.target.str.replace('\n|\r', '<n>')
    t.right = t.right.str.replace('\n|\r', '<n>')

    t = t.sort_values(by='target_len', ascending=False)

    test = t.target == t.target.str.lower()
    tsub = t.loc[test]
    for i, row in tsub.iterrows():
        print('\n', i, row.target, '|', row.left, '<TARGET>', row.right)

idx_rm_endo = [1096, 648]
m = matches_endo.loc[idx_rm_endo, ['left', 'target', 'right']]
for c in ['left', 'target', 'right']:
    m[c] = m[c].str.replace('\n|\r', '<n>')

for i, row in m.iterrows():
    print('\n----', i, row.left, '<<<', row.target, '>>>', row.right)
matches_endo.shape
matches_endo = matches_endo.loc[~matches_endo.index.isin(idx_rm_endo)]
matches_endo.shape

idx_rm_img = [404, 885, 104]
m = matches_img.loc[idx_rm_img, ['left', 'target', 'right']]
for c in ['left', 'target', 'right']:
    m[c] = m[c].str.replace('\n|\r', '<n>')

for i, row in m.iterrows():
    print('\n----', i, row.left, '<<<', row.target, '>>>', row.right)
matches_img.shape
matches_img = matches_img.loc[~matches_img.index.isin(idx_rm_img)]
matches_img.shape

idx_rm = [8606, 2861, 6820, 5704, 252, 8607]
m = matches.loc[idx_rm, ['left', 'target', 'right']]
for c in ['left', 'target', 'right']:
    m[c] = m[c].str.replace('\n|\r', '<n>')

for i, row in m.iterrows():
    print('\n----', i, row.target, '|', row.left, '<<<', row.target, '>>>', row.right)

matches.shape
matches = matches.loc[~matches.index.isin(idx_rm)]
matches.shape
#endregion

## Get maximum values per report
df, __ = get_tnm_values(df, matches=matches, col='unsafe_imaging_report')
df_img, __ = get_tnm_values(df_img, matches=matches_img, col='imaging_report')
df_endo, __ = get_tnm_values(df_endo, matches=matches_endo, col='report_text')


## Drop free text
df.columns
df = df.drop(labels=['source_pathology_id', 'surname', 'forename', 'unsafe_imaging_report',
                     'authorising_pathologist', 'reporting_pathologist'], axis=1)
df.columns

df_img.columns
df_img = df_img.drop(labels=['source_imaging_id', 'imaging_report', 'accession_number', 
                             'reported_by', 'typed_by', 'changed_by',
                             'verified_by', 'last_verify_by', 'last_updated', 'created'], axis=1)
df_img.columns

df_endo.columns
df_endo = df_endo.drop(labels=['report_text', 'report_template_id', 'report_id', 'create_date',
                               'true_start_time', 'true_end_time'], axis=1)
df_endo.columns

## Save to disk
df.to_csv(out_path / 'tnm_pathology.csv', index=False)
df_img.to_csv(out_path / 'tnm_radiology.csv', index=False)
df_endo.to_csv(out_path / 'tnm_endoscopy.csv', index=False)
