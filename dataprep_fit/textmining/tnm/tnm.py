"""High level function for extracting TNM phrases and TNM values from reports"""
from textmining.tnm.clean import filter_and_clean
from textmining.tnm.extract import tnm_phrase
from textmining.tnm.pattern import _tnm_value
from textmining.perineural import get_pn
from textmining.utils import extract
from dataclasses import asdict
import numpy as np
import pandas as pd
import regex as re
import warnings
import time
from joblib import Parallel, delayed
#import git

#try:
#    repo = git.Repo(search_parent_directories=True)
#    SHA = repo.head.object.hexsha
#except:
SHA = 'none'  # In case code is downloaded, not pulled


def get_tnm_phrase(df: pd.DataFrame, col: str, gapsize: int = 100, pad_left: int = 200, pad_right: int = 200,
                   extract_solitary: bool = True, flex_start: bool = False, simplicity: int = 2,
                   remove_unusual: bool = True, remove_historical: bool = False, remove_flex: bool = True,
                   remove_falsepos: bool = True):
    """Get all TNM phrases (single TNM values and TNM sequences)

    Args
        df: Pandas DataFrame that contains reports
        col: name of column in df that contains reports
        gapsize: maximum number of characters allowed between two consecutive TNM values when extracting TNM sequences
        pad_left, pad_right: number of characters to include on the left and right of each match,
                           used for visually inspecting results,
                           and for marking single TNM values for exclusion
        extract_solitary: additionally extract single TNM values (not contained in TNM sequences)
        flex_start: use a more flexible pattern for TNM sequences before applying a more constrained pattern
        simplicity: simplicity of TNM regex patterns (running time and accuracy trade-off) ...
        2 - most complex; requires specific sequences of TNM values; constrains context of individual values;
            allows a single variation in case, gap and mis-spelling of 0=O at a given time.
        1 - like 2, but allows any sequence of TNM values. Thus less complex.
        0 - simplest; allows any sequence of TNM values; does not constrain context; allows all variations in
            case, gap and mis-spelling of 0=O.
        remove_unusual: remove unusual TNM phrases from output (e.g. T-values that look like multiple choice prompts)
        remove_historical: remove historical TNM phrases from output
        remove_flex: remove TNM sequences that were obtained with too flexible patterns (applies if flex_start = True)
        remove_falsepos: remove single TNM-like values that have unwanted keywords nearby (such as 'T1 flare')

    Outputs
        matches: DataFrame with extracted TNM phrases, where 'target' contains cleaned TNM phrases
        check_phrases, check_cleaning, check_rm: additional dataframes for qualiy checks
            check_phrases: contains all unique extracted TNM phrases
            check_cleaning: shows TNM phrases before and after cleaning
            check_rm: shows TNM phrases marked for exclusion, and exclusion reason
    """

    # Get TNM phrases (sequences and solitary values)
    matches = tnm_phrase(df, col, gapsize, pad_left, pad_right, extract_solitary, flex_start, simplicity)

    # Remove phrases marked for exclusion?
    check_rm = pd.DataFrame()
    if matches.shape[0] > 0:

        if remove_flex:
            rm = matches.loc[(matches.exclusion_indicator == 1) & (matches.solitary_indicator == 0)]
            matches = matches.loc[~matches.index.isin(rm.index)]
            check_rm = pd.concat(objs=[check_rm, rm], axis=0)

        if remove_falsepos:
            rm = matches.loc[(matches.exclusion_indicator == 1) & (matches.solitary_indicator == 1)]
            matches = matches.loc[~matches.index.isin(rm.index)]
            check_rm = pd.concat(objs=[check_rm, rm], axis=0)

    # Additional filter and clean
    if matches.shape[0] > 0:
        flex_clean = True if simplicity == 0 else False  # Ensure values are cleaned with appropriately flex pattern
        matches, check_phrases, check_cleaning, rm = filter_and_clean(matches, clean=True, split_at_t=True,
                                                                      remove_unusual=remove_unusual,
                                                                      remove_historical=remove_historical,
                                                                      max_upper=100, max_digit=100,
                                                                      flex_clean=flex_clean)
        check_rm = pd.concat(objs=[check_rm, rm], axis=0)
    else:
        check_phrases, check_cleaning = pd.DataFrame(), pd.DataFrame()

    # Add identifier for git commit
    matches['sha'] = SHA

    return matches, check_phrases, check_cleaning, check_rm


def get_tnm_values(df: pd.DataFrame, matches: pd.DataFrame = None, col: str = 'report',
                   pathology_prefix: bool = False, combine_cats: bool = True, additional_output: bool = False):
    """Extract TNM values from TNM phrases, or straight from reports (not recommended)

    Args
        df: dataframe that contains reports
        matches: TNM phrases extracted from reports, outputted by get_tnm_phrase()
            If set to None, then each report is treated as phrase
        col: column of the dataframe that contains reports
        pathology_prefix: append letter 'p' in front of output columns, so they are pT, pN, pM etc
        combine_cats: if True, combine TNM values and subcategories into a single column
            If True, then column 'T' contains values '1', '1a', '1b', ... '2', '2a', ... etc
            If False, then column 'T' contains values '1', '2', '3', and 'T_sub' contains subcategories 'a', 'b' etc

    Output
        Original dataframe df with additional columns:
            T_pre - prefix for maximum T category value, such as 'p' or 'yp' etc
            T - max value for T category, such as '1', '1a'
            N, M, L, V, R, H, SM, Pn, G - max values for other TNM categories

            T_pre_min - prefix for minimum T category value
            T_min - minimum value for T category
            N_min, M_min, L_min, ... - min values for other TNM categories

        If additional_output is set to True, additional columns are included:
            T_indecision - equals 1 if T and T_min values differ
            T_pre_indecision, N_indecision, M_indecision, ... - indecision flag for other TNM categories

            yTNM, rTNM - whether prefix for maximum T value contains y or r
            yTNM_min, rTNM_min - whether prefix for minimum T value contains y or r
            yTNM_any, rTNM_any - whether max or min T value prefix contains y or r
    """
    tic = time.time()

    # Check if dataframe already has columns that are to be added
    df = _check_val_cols(df)

    # If matches is not provided, assume values need to be extracted directly from reports
    if matches is None:
        matches = df.copy()
        matches['row'] = np.arange(matches.shape[0])
        matches = matches.rename(columns={col: 'target'})
        matches['length'] = matches.target.str.len()

    # Map original reports ('row') to extracted targets
    # Retain only unique targets to speed up subsequent extraction
    targetmap = matches[['row', 'target']].drop_duplicates()
    matches = matches[['target', 'length']].copy().drop_duplicates()

    # ==== Get values from the phrase, e.g. T1 from 'T1 N0 V0 L0 M1' ====
    print('\nExtracting values from the phrase ...')

    # ---- (1) Extract letters together with values: 'T1', 'T1a/2/3', 'T1-3', etc ----
    submatches = _extract_val(matches)

    # ---- (2) Then get individual values, e.g. 1, a, 2, 3 from T1a/2/3 ----
    pat = '([O01234xX]|is)([abcdABCD]{,1})'
    subsubmatches = extract(submatches, 'values', pat=pat, flags=re.DOTALL)
    subsubmatches = subsubmatches.rename(columns={'target': 'value', 'row': 'row_match',
                                                  'target_group1': 'value_main', 'target_group2': 'value_sub'})

    # ---- Combine (1) and (2) ----
    submatches['row_match'] = np.arange(submatches.shape[0])
    submatches = submatches.merge(subsubmatches[['row_match', 'value', 'value_main', 'value_sub']],
                                  how='left', on='row_match').drop('row_match', axis=1)

    # ---- Clean ----
    #  T and N subcategories to lowercase (e.g. 'A' in 'T1A' to 'a')
    #  x to uppercase, O to 0, IS to lowercase
    submatches.value_sub = submatches.value_sub.str.lower()
    submatches.value_main = submatches.value_main.str.upper()
    submatches.value_main = submatches.value_main.str.replace('IS', 'is', regex=True, flags=re.IGNORECASE)
    submatches.value_main = submatches.value_main.str.replace('O', '0', regex=True, flags=re.IGNORECASE)
    submatches['prefix'] = submatches['prefix'].str.lower()
    submatches['prefix'] = pd.Series([''.join(sorted(s)) if not pd.isna(s) else s for s in submatches['prefix'].copy()])

    # ---- Map values back to reports ----
    submatches = submatches.merge(targetmap, how='left', on=['target'])

    # ---- Get additional matches for peri-neural invasion ----
    _, x = get_pn(df.copy(), col, varname='Pn')  # Need df.copy() atm as get_pn will create min-max cols for df
    x = x.loc[x.value != '']
    x['value_main'] = x['value'].copy()
    x = x.drop(labels=['pat_id', 'sentence', 'sentence_left', 'sentence_right'], axis=1)
    x = x.rename(columns={'variable': 'tnm_category'})
    submatches = pd.concat(objs=[submatches, x], axis=0)

    # ==== Get minimum and maximum values ====
    if submatches.shape[0] > 0:
        p = _extract_val_minmax(submatches)
    else:
        p = pd.DataFrame()

    # ==== Tidy up columns ====
    #if report_type == 'imaging':
    #    cols_max = ['T_pre', 'T', 'T_sub', 'N', 'N_sub', 'M', 'M_sub', 'V', 'R', 'SM', 'H']
    cols_max = ['T_pre', 'T', 'T_sub', 'N', 'N_sub', 'M', 'M_sub', 'V', 'R', 'L', 'Pn', 'SM', 'H', 'G']
    cols_min = [c + '_min' for c in cols_max]
    cols_ind = [c + '_indecision' for c in cols_max]
    cols = ['row'] + cols_max + cols_min + cols_ind + ['yTNM', 'rTNM', 'yTNM_min', 'rTNM_min', 'yTNM_any', 'rTNM_any']
    cols = np.array(cols)

    # If a variable was not extracted, add an empty column
    for c in cols:
        if np.isin([c], p.columns) == False:
            p[c] = np.nan
    p = p[cols]

    # Simplify output
    if combine_cats:
        pairs = [['T', 'T_sub'], ['N', 'N_sub'], ['M', 'M_sub']]
        for pair in pairs:
            p[pair[0]] = p[pair[0]] + p[pair[1]].fillna('')
            p = p.drop(labels=[pair[1]], axis=1)

            p[pair[0] + '_min'] = p[pair[0] + '_min'] + p[pair[1] + '_min'].fillna('')
            p = p.drop(labels=[pair[1] + '_min'], axis=1)

    if not additional_output:
        col_drop = cols_ind + ['yTNM', 'rTNM', 'yTNM_min', 'rTNM_min', 'yTNM_any', 'rTNM_any']
        p = p.drop(col_drop, axis=1)

    # Add p prefix to TNM column names if requested
    if pathology_prefix:
        cols = p.columns
        mask = cols.str.contains('^[TNM]', flags=re.DOTALL)
        cols = cols.to_numpy()
        cols[mask] = 'p' + cols[mask]
        p.columns = cols

    # Add to original dataframe
    df_return = df.copy()
    df_return['row'] = np.arange(df.shape[0])
    df_return = df_return.merge(p, how='left', on='row')
    df_return = df_return.drop(['row'], axis=1)

    # Add identifier for git commit
    df_return['sha'] = SHA

    toc = time.time()
    print('Time elapsed: {:.2f} minutes'.format((toc - tic) / 60))

    return df_return, submatches


def _check_val_cols(df):
    """Check if dataframe already has columns for the extracted variables"""
    cols_max = ['T_pre', 'T', 'T_sub', 'N', 'N_sub', 'M', 'M_sub', 'V', 'R', 'L', 'Pn', 'SM', 'H', 'G']
    cols_min = [c + '_min' for c in cols_max]
    cols_ind = [c + '_indecision' for c in cols_max]
    cols_all = ['row'] + cols_max + cols_min + cols_ind + ['yTNM', 'rTNM', 'yTNM_min', 'rTNM_min', 'yTNM_any',
                                                           'rTNM_any']
    cols_all = np.array(cols_all)
    test = np.isin(cols_all, df.columns)
    if test.any():
        warnings.warn(
            'Some columns in input data have the same name as columns that will be added: these will first be removed')
        df = df.drop(cols_all[test], axis=1)
    return df


def _extract_val(matches, pn_sensitive=True):
    """Extract TNM values from TNM phrases
    pn_sensitive flag is for extracting values in a way that better distinguishes N and Pn,
    but it is potentially more conservative.
    Without it, the N value in phrases like 'T0 N0 Pn1' will be mistakenly set as 1.
    """

    # Get flexible value patterns (assuming false matches have been cleaned out)
    pats = _tnm_value(constrain_context=True)
    pats = asdict(pats)
    if pn_sensitive:
        pdict = {}
        for key, val in pats.items():
            if key in ['Pn', 'N']:
                pdict[key] = [val[k] for k in ['norm', 'gap', 'case', 'mis']]
            else:
                pdict[key] = val['flex']
    else:
        pdict = {key: val['flex'] for key, val in pats.items()}


    # If code is run on imaging reports, do not extract all variables
    #if report_type == 'imaging':
    #    vars_imaging = ['T', 'N', 'M', 'L', 'R', 'SM', 'H']
    #    pdict = {key: val for key, val in pdict.items() if np.isin(key, vars_imaging)}

    # Extract
    submatches = pd.DataFrame()
    if pn_sensitive:
        for key, val in pdict.items():
            if key in ['Pn', 'N']:
                for subval in val:
                    m = extract(matches, 'target', subval, flags=re.DOTALL)
                    m['tnm_category'] = key
                    if m.shape[0] > 0:
                        submatches = pd.concat(objs=[submatches, m], axis=0)
                        continue
            else:
                m = extract(matches, 'target', val, flags=re.DOTALL)
                m['tnm_category'] = key
                submatches = pd.concat(objs=[submatches, m], axis=0)
    else:
        for key, val in pdict.items():
            m = extract(matches, 'target', val, flags=re.DOTALL)
            m['tnm_category'] = key
            submatches = pd.concat(objs=[submatches, m], axis=0)        
    submatches = submatches.rename(columns={'target': 'letter_and_values', 'target_group1': 'prefix',
                                            'target_group2': 'letter', 'target_group3': 'values'})

    # Add original target back to submatches
    matches['row_match'] = np.arange(matches.shape[0])
    submatches['row_match'] = submatches.row.copy()
    submatches = submatches.merge(matches[['target', 'row_match']], how='left', on='row_match')
    submatches = submatches.drop(['row', 'row_match'], axis=1)

    # Reorder columns
    submatches = submatches[['target', 'start', 'end', 'left', 'letter_and_values', 'right', 'prefix', 'letter',
                             'values', 'tnm_category']]
    return submatches


def _extract_val_minmax(submatches):
    print('\nGetting minimum and maximum values ...')

    # Create a temporary variable where 'X' is replaced with '01' and 'is' with '02', to allow min-max sorting
    submatches['value_tmp'] = submatches['value'].copy()
    submatches['value_tmp'] = submatches['value_tmp'].str.lower().replace({'x': '01', 'is': '02'})
    submatches['let_val_len'] = submatches['letter_and_values'].fillna('').apply(len)

    # Sort by tnm category, row (=report) and temporary variable in descending order
    # Maximum values of the variable for each variable-row combination are in the first rows
    # Minimum values of the variable for each variable-row combination are in the last rows
    # Sorting by let_val_len just in case multiple matches incl for same TNM value, then including largest
    submatches = submatches.sort_values(['tnm_category', 'row', 'value_tmp', 'let_val_len'], ascending=False)
    smax = submatches.groupby(['tnm_category', 'row'], as_index=False).first()
    smin = submatches.groupby(['tnm_category', 'row'], as_index=False).last()

    # Retain only some columns and rename
    cols = ['row', 'tnm_category', 'value', 'value_main', 'value_sub', 'prefix']
    cols_min = {c: c + '_min' for c in cols if not np.isin(c, ['row', 'tnm_category'])}
    cols_max = {c: c + '_max' for c in cols if not np.isin(c, ['row', 'tnm_category'])}
    smin = smin[cols].rename(columns=cols_min)
    smax = smax[cols].rename(columns=cols_max)

    # Pivot to get maximum TNM staging, subdivisions for T and N stage, and prefix of T in separate columns
    # Note that maximum values won't have a suffix '_max', whereas minimum values do
    # Also note that 'value_main' contains bare values (e.g. 1 in pT1a); prefix has 'T' and 'value_sub' has 'a'
    pmax = smax.pivot(index='row', columns='tnm_category', values='value_main_max')
    pmax2 = smax.loc[smax.tnm_category.isin(['T', 'N', 'M'])].pivot(index='row', columns='tnm_category',
                                                                values='value_sub_max')
    pmax2.columns = pmax2.columns + '_sub'
    pmax3 = smax.loc[smax.tnm_category.isin(['T'])].pivot(index='row', columns='tnm_category', values='prefix_max')
    pmax3.columns = pmax3.columns + '_pre'

    # Pivot to get minimum TNM staging and subdivisions for T and N stage in separate columns
    # Add '_min' suffix to distiguish from maximum values
    pmin = smin.pivot(index='row', columns='tnm_category', values='value_main_min')
    pmin.columns = pmin.columns + '_min'

    pmin2 = smin.loc[smin.tnm_category.isin(['T', 'N', 'M'])].pivot(index='row', columns='tnm_category',
                                                                values='value_sub_min')
    pmin2.columns = pmin2.columns + '_sub_min'

    pmin3 = smin.loc[smin.tnm_category.isin(['T'])].pivot(index='row', columns='tnm_category', values='prefix_min')
    pmin3.columns = pmin3.columns + '_pre_min'

    # Combine minimum and maximum values for each report to create an indecision flag
    #  Note: indecision is based on category plus subcategory, e.g. T3 vs T3a -> indecision
    #  Note: previously, there was a bug, where case difference caused indecision, e.g. x != X
    sind = pd.merge(smin, smax, how='inner', on=['row', 'tnm_category'])
    sind['indecision'] = 0
    sind.loc[sind.value_min.str.upper() != sind.value_max.str.upper(), 'indecision'] = 1
    pind = sind.pivot(index='row', columns='tnm_category', values='indecision')
    pind.columns = pind.columns + '_indecision'

    # Combine mininum, maximum and indecision columns for each report
    p = pmax.merge(pmax2, how='outer', left_index=True, right_index=True) \
        .merge(pmax3, how='outer', left_index=True, right_index=True) \
        .merge(pmin, how='outer', left_index=True, right_index=True) \
        .merge(pmin2, how='outer', left_index=True, right_index=True) \
        .merge(pmin3, how='outer', left_index=True, right_index=True) \
        .merge(pind, how='outer', left_index=True, right_index=True)

    # Replace empty values with nan
    p = p.astype('object')
    p = p.replace('', np.nan)

    # Pull the 'row' variable out of index (row = report)
    p = p.reset_index()
    p.columns.name = None

    # Ensure prefix is in string format
    for c in ['T_pre', 'T_pre_min']:
        if ~np.isin(c, p.columns):
            p[c] = np.nan
    p.T_pre = p.T_pre.replace(np.nan, '')
    p.T_pre_min = p.T_pre_min.replace(np.nan, '')

    # Extract whether T stage has y or r prefix -- for maximum T stage value
    p['yTNM'] = 0
    mask = p.T_pre.str.contains('y', flags=re.IGNORECASE)
    mask = mask.fillna(False)
    p.loc[mask, 'yTNM'] = 1

    p['rTNM'] = 0
    mask = p.T_pre.str.contains('r', flags=re.IGNORECASE)
    mask = mask.fillna(False)
    p.loc[mask, 'rTNM'] = 1

    # Extract whether T stage has y or r prefix -- for minimum T stage value
    p['yTNM_min'] = 0
    mask = p.T_pre_min.str.contains('y', flags=re.IGNORECASE)
    mask = mask.fillna(False)
    p.loc[mask, 'yTNM_min'] = 1

    p['rTNM_min'] = 0
    mask = p.T_pre_min.str.contains('r', flags=re.IGNORECASE)
    mask = mask.fillna(False)
    p.loc[mask, 'rTNM_min'] = 1

    # Convert empty prefixes to nan again
    p.T_pre = p.T_pre.replace('', np.nan)
    p.T_pre_min = p.T_pre_min.replace('', np.nan)

    # Extract whether T stage has y or r prefix -- for any T stage value
    # NB select MAX value per row, o.w. repeat rows
    s = submatches[['row', 'prefix']].drop_duplicates().copy()
    s['yTNM_any'] = 0
    mask = s.prefix.fillna('').str.contains('y', flags=re.IGNORECASE)
    s.loc[mask, 'yTNM_any'] = 1
    s = s.groupby('row', as_index=False)['yTNM_any'].any().astype(int)
    p = p.merge(s[['row', 'yTNM_any']], how='left', on='row')

    s = submatches[['row', 'prefix']].drop_duplicates().copy()
    s['rTNM_any'] = 0
    mask = s.prefix.fillna('').str.contains('r', flags=re.IGNORECASE)
    s.loc[mask, 'rTNM_any'] = 1
    s = s.groupby('row', as_index=False)['rTNM_any'].any().astype(int)
    p = p.merge(s[['row', 'rTNM_any']], how='left', on='row')

    return p


def get_tnm_phrase_par(nchunks, njobs, df, col, **kwargs):
    """Runs the get_tnm_phrase function in parallel"""
    tic = time.time()
    
    test = df.shape[0] == df.index.nunique()
    if not test:
        raise ValueError("Index of df is not unique; run 'df.reset_index(drop=True)' first")
    
    indices = np.arange(df.shape[0])

    def _process_chunk(indices):
        dfsub = df.iloc[indices]
        m, __, __, __ = get_tnm_phrase(df=dfsub, col=col, **kwargs)
        row_map = {i:row for i, row in enumerate(indices)}
        m['row'] = m.row.replace(row_map)
        return m

    if nchunks == 1:
        out = _process_chunk(df.index)
        out = [out]
    else:
        chunks = np.array_split(indices, nchunks)
        out = Parallel(n_jobs=njobs)(delayed(_process_chunk)(indices) for indices in chunks)
    
    matches = pd.concat(objs=out, axis=0)
    matches = matches.sort_values(by=['row', 'start', 'end', 'target']).reset_index(drop=True)
    
    toc = time.time()
    print('Time elapsed: {} minutes'.format((toc - tic) / 60))

    return matches
