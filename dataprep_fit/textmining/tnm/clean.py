from textmining.constants import VOCAB_DIR
from textmining.reports import get_crc_reports
from textmining.tnm.pattern import _tnm_value
from textmining.utils import extract, extract_set, process, get_sentence, get_context_patterns, wrap_pat, constrain
from dataclasses import asdict
import pandas as pd
import numpy as np
import regex as re
import time

CTX_FILE = 'context_tnm.csv'


# ==== For filtering and cleaning TNM phrases ====
def filter_and_clean(matches, clean=True, split_at_t=True, remove_unusual=True, remove_historical=False,
                     max_upper=100, max_digit=100, flex_clean=False):
    """Filters and cleans TNM phrases

        clean: clean TNM phrases, e.g. by removing text between TNM values, correcting unusually written values, ...
        split_at_t: split long TNM phrases at T-value, e.g. if a single TNM phrase contains multiple sub-phrases
        remove_unusual: remove unusual TNM phrases from output (e.g. T-values that look like multiple choice prompts)
        remove_historical: remove historical TNM phrases from output
        max_upper, max_digit: see filter_words()
        flex_clean: if True, uses flexible TNM pattern for retaining TNM-like parts
                    if False (default), uses a TNM pattern that allows for one deviation for normal at a time
    """

    # Find TNM phrases with unusually large number of uppercase letters and digits (and optionally remove)
    matches, check_rm0 = filter_words(matches, max_upper=max_upper, max_digit=max_digit, remove=remove_unusual)

    # Split a TNM phrase into multiple TNM phrases at each T value
    # E.g. split "pT1 N0 M0 txt pT2 N1 MX" into "pT1 N0 M0 txt" and "pT2 N1 MX"
    if split_at_t:
        pats = _tnm_value(constrain_context=True)
        tx = pats.T.comb
        matches = str_split(matches, 'target', pat=tx)
        print('Number of matches after splitting phrases: {}'.format(matches.shape[0]))
    else:
        matches['target_before_split'] = matches['target']

    # Find TNM phrases that have historical staging (and optionally remove)
    matches, check_rm1 = filter_prev(matches, remove=remove_historical, use_context=True)

    # Get dataframe for TNM phrases that were marked for exclusion (and optionally removed)
    check_rm = pd.concat(objs=[check_rm0, check_rm1], axis=0)
    check_rm = check_rm[['row', 'left', 'target', 'right', 'exclusion_indicator', 'exclusion_reason']]

    # Clean phrases
    #  (1) to reduce the number of unique phrases where information needs to be extracted, and
    #  (2) to remove some false positives
    if clean:
        matches = clean_tnm(matches, target_col='target', flex_clean=flex_clean)
        matches['length'] = matches.target.str.len()
        check_cleaning = matches.loc[matches.exclusion_indicator == 0, ['target_before_clean', 'target', 'length']]
        check_cleaning = check_cleaning.copy().drop_duplicates().sort_values(by='length', ascending=False)
    else:
        matches['target_before_clean'] = matches['target']
        check_cleaning = None

    # Compute length, and extract unique phrases for quality checks
    matches['length'] = matches.target.str.len()
    matches = matches.sort_values('length', ascending=False)
    check_phrases = matches.loc[matches.exclusion_indicator == 0, ['target', 'length']].copy().drop_duplicates()

    # Reset index and sort
    matches = matches.reset_index(drop=True)
    matches = matches.sort_values(by=['row', 'start'], ascending=[True, True])

    return matches, check_phrases, check_cleaning, check_rm


def str_split(df, col, pat='pT', sort_cols=['row']):
    """Splits a long TNM phrase into multiple phrases based on pattern pat
    For example, if pat matches for T values, then 'T1 N0 M0 text T2 N0 M0' is split into 'T1 N0 M0 text' and 'T2 N0 M0'
    """
    # print(df.shape)
    # Temporary variable that represents the rows of input df
    df['tmp_row'] = np.arange(df.shape[0])

    # Find matches for pattern pat in column col in dataframe df
    # E.g. find T-stage values within all previously detected TNM  phrases, if df is the dataframe of TNM phrases
    m = extract(df, col, pat, groups=False, flags=re.DOTALL)
    m = m.reset_index(drop=True).rename(
        columns={'row': 'tmp_row', 'start': 'split_start'})  # rename to avoid overlap with df
    m = m[['tmp_row', 'split_start']].merge(df, how='left', on=['tmp_row'])  # merge original dataframe back

    # Computer number of matches per row (if number = 1, no split needed)
    nmatch = m.groupby('tmp_row').size().rename('tmp_nmatch').reset_index()
    m = m.merge(nmatch, how='left')

    if nmatch['tmp_nmatch'].max() > 1:
        m['target_before_split'] = m[col].copy()
        m['split_indicator'] = 0

        # Gather rows that did not match at all, as these need retaining
        idx_drop = df.iloc[m.tmp_row.drop_duplicates(), :].index  # Index for matches
        m_no = df.drop(idx_drop, axis=0)  # Drop rows from df with index for matches
        m_no['target_before_split'] = m_no[col].copy()
        m_no['split_indicator'] = 0

        # Separate out rows that matched only once (these won't be split)
        m_out = m.loc[m.tmp_nmatch == 1].copy()
        m = m.loc[m.tmp_nmatch > 1]
        m['split_indicator'] = 1

        # Split indices
        m['split_end'] = m['split_start'].shift(-1)
        m['tmp_length'] = m[col].apply(len)
        idx = m.groupby('tmp_row').tail(1).index
        m.loc[idx, 'split_end'] = m.loc[idx, 'tmp_length']
        m['split_end'] = m['split_end'].astype(int)
        idx = m.groupby('tmp_row').tail(0).index
        m.loc[idx, 'split_start'] = 0  # Start first match from 0

        # Split
        m['substring'] = m.apply(lambda x: x[col][x['split_start']:x['split_end']], axis=1)

        # Add or correct text to the left and right of splits
        left = m.apply(lambda x: x[col][0:x['split_start']], axis=1)
        right = m.apply(lambda x: x[col][x['split_end']:x['tmp_length']], axis=1)
        if 'left' in m.columns and 'right' in m.columns:
            m['left_before_split'] = m['left'].copy()
            m['right_before_split'] = m['right'].copy()
            m['left'] = m['left'] + left
            m['right'] = right + m['right']
        else:
            m['left'] = left
            m['right'] = right

        # Correct start and end positions of splits
        if np.isin(['start', 'end'], df.columns).all():
            mask = ~m.split_end.isna()
            m.loc[mask, 'end'] = m.loc[mask, 'start'] + m.loc[mask, 'split_end']
            mask = m.split_start > 0
            m.loc[mask, 'start'] = m.loc[mask, 'start'] + m.loc[mask, 'split_start']

        # Replace original column with splitted
        m[col] = m['substring']
        m = m.drop(labels='substring', axis=1)

        # Concatenate and remove temporary cols
        m = pd.concat(objs=[m, m_out, m_no], axis=0)
        cols = [c for c in m.columns if not c.startswith('tmp_')]
        m = m[cols]

        # Sort rows and cols
        m = m.sort_values(sort_cols + ['split_start'])
        newcols = sort_cols + ['split_indicator', 'split_start', 'split_end', 'target_before_split', 'left', col,
                               'right']
        col_order = newcols + [c for c in m.columns if c not in newcols]
        m = m[col_order]

        # Fillna
        m[['left', 'right']] = m[['left', 'right']].fillna('')
        # print(m.shape)

        # Ensure int dtype
        if np.isin(['start', 'end'], df.columns).all():
            m.start = m.start.astype(int)
            m.end = m.end.astype(int)
    else:
        m = df.copy()
        m['target_before_split'] = m[col].copy()
    return m


def filter_words(matches, max_upper=40, max_digit=15, remove=False):
    """
     Removes some potential false positives
      number of uppercase letters is greater than max_upper
      number of digits is greater than max_digits
      has more than 4 T-stage values -- likely unused multiple choice
    """
    matches = matches.reset_index(drop=True)

    if 'exclusion_indicator' not in matches.columns:
        matches['exclusion_indicator'] = 0
    if 'exclusion_reason' not in matches.columns:
        matches['exclusion_reason'] = ''
    matches['tmp_ind'] = 0

    # Uppercase letters
    matches['n_upper'] = matches['target'].str.count('[A-Z]')
    mask = matches.n_upper >= max_upper
    matches.loc[mask, ['exclusion_indicator', 'tmp_ind']] = 1
    matches.loc[mask, 'exclusion_reason'] += 'more than ' + str(max_upper) + ' uppercase letters;'
    print('{} matches have at least {} capital letters'.format(mask.sum(), max_upper))

    # Digits
    matches['n_digit'] = matches['target'].str.count('\d')
    mask = matches.n_digit >= max_digit
    matches.loc[mask, ['exclusion_indicator', 'tmp_ind']] = 1
    matches.loc[mask, 'exclusion_reason'] += 'more than ' + str(max_digit) + ' digits;'
    print('{} matches have at least {} digits'.format(mask.sum(), max_digit))

    # T stage values
    pats = _tnm_value(constrain_context=True)
    tx = pats.T.comb
    m = process(matches, 'target', [(tx + '.{,}?') * 4], action='keep', flags=re.DOTALL)
    mask = matches.index.isin(m.index)
    matches.loc[mask, ['exclusion_indicator', 'tmp_ind']] = 1
    matches.loc[mask, 'exclusion_reason'] += '4 or more T-values;'
    print('{} matches have at least 4 T values in a sequence'.format(mask.sum(), max_digit))

    # For checking matches that are marked for removal
    check = matches.loc[matches.tmp_ind == 1]
    if remove:
        matches = matches.loc[matches.tmp_ind == 0]
    matches = matches.drop(labels='tmp_ind', axis=1)
    return matches, check


def filter_rank(matches, keep_rule='keep_first', remove=False):
    """Marks matches for inclusion if they are the first or last in a report"""
    if 'exclusion_indicator' not in matches.columns:
        matches['exclusion_indicator'] = 0
    if 'exclusion_reason' not in matches.columns:
        matches['exclusion_reason'] = ''
    matches['tmp_ind'] = 0

    # Rank matches based on position
    ranks = matches.groupby(['row'], as_index=False)['start'].rank().rename(columns={'start': 'rank'})
    matches = matches.merge(ranks, how='left', left_index=True, right_index=True)

    # Mark for exclusion
    if keep_rule == 'keep_first':
        idx = matches.groupby('row')['rank'].idxmin()
        r = 'not the first match;'
    elif keep_rule == 'keep_last':
        idx = matches.groupby('row')['rank'].idxmax()
        r = 'not the last match;'
    tmp = matches.loc[idx].copy()
    mask = ~matches.index.isin(tmp.index)
    matches.loc[mask, ['exclusion_indicator', 'tmp_ind']] = 1
    matches.loc[mask, 'exclusion_reason'] += r

    # For checking matches that are marked for removal
    check = matches.loc[matches.tmp_ind == 1]
    if remove:
        matches = matches.loc[matches.tmp_ind == 0]
    matches = matches.drop(labels='tmp_ind', axis=1)
    return matches, check


def filter_prev(matches, remove=False, use_context=True, vocab_path=VOCAB_DIR):
    """Marks TNM phrases for removal, if nearby words indicate that these refer to historical TNM staging"""
    matches = matches.reset_index(drop=True)

    if 'exclusion_indicator' not in matches.columns:
        matches['exclusion_indicator'] = 0
    if 'exclusion_reason' not in matches.columns:
        matches['exclusion_reason'] = ''
    matches['tmp_ind'] = 0

    if 'sentence_left' not in matches.columns:
        matches = get_sentence(matches)

    if use_context:

        # Apply context-like algorithm
        vcon = pd.read_csv(vocab_path / CTX_FILE)
        vcon = vcon.replace({'': np.nan})
        char = r'[ \w\t\:\,\-\(\[\)\]]'
        pdist = None  # use pdist given in context.csv
        tdist = 100
        hist_left = get_context_patterns(vcon, category='historic', side='left', char=char, pdist=pdist, tdist=tdist,
                                         verbose=False)
        hist_right = get_context_patterns(vcon, category='historic', side='right', char=char, pdist=pdist, tdist=tdist,
                                          verbose=False)
        # display(hist_left)
        # display(hist_right)

        flags = re.IGNORECASE | re.DOTALL
        m = process(matches, 'left', [hist_left], action='remove', flags=flags) if hist_left else matches
        m = process(m, 'right', [hist_right], action='remove', flags=flags) if hist_right else m
        mask = ~matches.index.isin(m.index)
        matches.loc[mask, ['exclusion_indicator', 'tmp_ind']] = 1
        matches.loc[mask, 'exclusion_reason'] += 'refers to previously given staging;'

    else:
        # ==== Process left side ====

        # Patterns to look for from the left side, together with distance
        pats_rm = {'previ[\w ]{1,10} stag[ie]': 60,  # previous staging, previously staged as
                   'previous': 60,
                   'compar[ie]': 60,
                   'known': 60,
                   'prior': 60,
                   }
        pats_rm = [constrain([p], side='left', pat_type='wordstart', char='.', distance=d)[0] for p, d in
                   pats_rm.items()]

        # Patterns to keep from the left side
        pats_keep = {
            r'(based on|together|incorpor|includ|taking|conjunction|as from|given|final|summary|conclusion).{,50}previous': 60}
        pats_keep = [constrain([p], side='left', pat_type=None, char='.', distance=d)[0] for p, d in pats_keep.items()]

        # Remove matches that match for pats_rm but not for pats_keep in 'sentence_left' column
        m = process(matches, 'sentence_left', pats_rm, action='keep')
        m = process(m, 'sentence_left', pats_keep, action='remove')
        mask = matches.index.isin(m.index)
        matches.loc[mask, ['exclusion_indicator', 'tmp_ind']] = 1
        matches.loc[mask, 'exclusion_reason'] += 'refers to previously given staging (words on left);'

        # ==== Process right side ====

        # Patterns to look for from the right side
        pats_rm = {r'previ[\w ]{1,10} stag[ie]': 60,  # previous staging, previously staged
                   # 'based on[\w ]{,10} previ':50,  # Based on previous staging
                   'previous': 60
                   }
        pats_rm = [constrain([p], side='right', pat_type='wordstart', char='.', distance=d)[0] for p, d in
                   pats_rm.items()]

        # Patterns to keep from the right side
        pats_keep = {'retriev.{,30}in previ': 50,
                     r'(based on|together|incorpor|includ|taking|conjunction|as from|given|final|summary|conclusion).{,50}previous': 50}
        pats_keep = [constrain([p], side='right', pat_type=None, char='.', distance=d)[0] for p, d in pats_keep.items()]

        # Remove matches that match for pats_rm but not for pats_keep in 'sentence_right' column
        m = process(matches, 'sentence_right', pats_rm, action='keep')
        m = process(m, 'sentence_right', pats_keep, action='remove')
        mask = matches.index.isin(m.index)
        matches.loc[mask, ['exclusion_indicator', 'tmp_ind']] = 1
        matches.loc[mask, 'exclusion_reason'] += 'refers to previously given staging (words on right);'

    # For checking matches that are marked for removal
    check = matches.loc[matches.tmp_ind == 1]
    if remove:
        matches = matches.loc[matches.tmp_ind == 0]
    matches = matches.drop(labels='tmp_ind', axis=1)
    return matches, check


# Cleaning the TNM phrase
def clean_tnm(matches, target_col='target', flex_clean=False):
    """
    Removes text between TNM values
    Replaces some nonword characters
    Corrects the letters of a few TNM values, e.g. replaces Lv with L
    """
    # Create a copy of target column
    matches[target_col + '_before_clean'] = matches[target_col].copy()

    # Get patterns for extracting TNM values
    pats = _tnm_value(constrain_context=True)
    pats_dict = asdict(pats)
    if flex_clean:
        patlist = [pats_dict[key]['flex'] for key in pats_dict.keys()]
    else:
        patlist = [pats_dict[key]['comb'] for key in pats_dict.keys()]
    pat = wrap_pat(patlist)

    # Only retain that part of the string that matches for TNM value patterns
    matches = matches.reset_index(drop=True)
    m = extract_set(matches, target_col, [pat], flags=re.DOTALL, groups=False)
    m = m.sort_values(['row', 'start'])
    m = m.groupby('row')['target'].apply(' '.join).rename(target_col)  # Rename it is as target_col
    matches = matches.drop(target_col, axis=1)
    matches = matches.merge(m, how='left', left_index=True, right_index=True)

    # Remove some nonword characters
    matches[target_col] = matches[target_col].fillna('').astype(str)
    matches[target_col] = matches[target_col].str.replace(r'[\(\[]y[\)\]]', 'y', regex=True,
                                                          flags=re.DOTALL)  # Special case for (y)
    matches[target_col] = matches[target_col].str.replace(r'[\(\[\)\]]', ' ', regex=True, flags=re.DOTALL)
    matches[target_col] = matches[target_col].str.replace(r'[\r\n\,\.\+\_]', ' ', regex=True, flags=re.DOTALL)
    matches[target_col] = matches[target_col].str.replace(' {2,}', ' ', regex=True, flags=re.DOTALL)
    matches[target_col] = matches[target_col].str.replace(' *$', '', regex=True)
    matches[target_col] = matches[target_col].str.replace('^ *', '', regex=True)

    # Replace roman numbers with value
    matches[target_col] = matches[target_col].str.replace(r'\bIV\b', '4', regex=True, flags=re.DOTALL|re.IGNORECASE)
    matches[target_col] = matches[target_col].str.replace(r'\bIII\b', '3', regex=True, flags=re.DOTALL|re.IGNORECASE)
    matches[target_col] = matches[target_col].str.replace(r'\bII\b', '2', regex=True, flags=re.DOTALL|re.IGNORECASE)
    matches[target_col] = matches[target_col].str.replace(r'\bI\b', '1', regex=True, flags=re.DOTALL|re.IGNORECASE)

    # Replace Lv or Ly with L, e.g. Lv1 = L1. Ignoring case
    pats = [r'Lv {,1}(\d)', r'Ly {,1}(\d)']
    for pat in pats:
        matches[target_col] = matches[target_col].str.replace(pat, r'L\g<1>', regex=True,
                                                              flags=re.DOTALL | re.IGNORECASE)

    # Replace uncommon patterns such as T4a & b with T4a/4b
    for i in range(3):
        matches[target_col] = matches[target_col].str.replace(r'(\d)([a-d]) {,2}\W {,2}(?:\bor\b {,2})?([a-d])',
                                                              r'\g<1>\g<2>/\g<1>\g<3>',
                                                              regex=True, flags=re.IGNORECASE | re.DOTALL)

    # Replace 'kikuchi level sm...' with 'sm...'
    matches[target_col] = matches[target_col].str.replace(r'kikuchi\s(?:level)?\W{,3}(?=sm)', '', regex=True,
                                                          flags=re.IGNORECASE)
    return matches


# ==== For finding nearby tumour keywords ====
def add_tumour_tnm(df, matches, col_report, pad_left=50, pad_right=50, mark_left='<<', mark_right='>>', crc_only=False,
                   targetcol='target'):
    print('Finding nearby tumour keywords for each TNM phrase')
    tic = time.time()
    matches = matches.copy()

    # Get matches for tumour keywords
    _, m = get_crc_reports(df, col_report, verbose=False, pad_left=pad_left, pad_right=pad_right)
    if crc_only:
        m = m.loc[m.exclusion_indicator == 0]
    print('  Number of matches for tumour keywords that were included: {}'.format(m.shape[0]))
    m = m.rename(columns={'target': targetcol})  # For consistency with matches_to_phrase()

    # Retain matches in reports that also have TNM value
    m = m.loc[m.row.isin(matches.row)]

    # Combine with matches for TNM keywords
    m['tum_ind'] = 1
    matches['tum_ind'] = 0
    m = pd.concat(objs=[matches, m], axis=0).sort_values(by=['row', 'start'], ascending=[True, True])
    m = m.reset_index(drop=True)

    # For each TNM keyword, get nearest tumour keyword on left and right
    idx = m.loc[m.tum_ind == 0].index
    mtum = m.loc[m.tum_ind == 1]
    res = pd.DataFrame()
    for i in idx:
        i = np.array([i])  # Index for TNM keyword
        row = m.loc[i, 'row'].iloc[0]  # Report that the TNM keyword occurs in

        idx_left = np.array(i - 1)
        idx_left = idx_left[np.isin(idx_left, mtum.loc[mtum.row == row].index)]
        idx_right = np.array(i + 1)
        idx_right = idx_right[np.isin(idx_right, mtum.loc[mtum.row == row].index)]

        idx_keep = np.concatenate([idx_left, i, idx_right])
        msub = m.loc[idx_keep]
        # display(msub)

        t = m.loc[i].reset_index(drop=True)
        r, _ = matches_to_phrase(msub, targetcol=targetcol, colname='phrase_with_tumour', mark_left=mark_left,
                                 mark_right=mark_right)
        # display(r)
        r = r.drop(labels='row', axis=1)
        r = pd.concat(objs=[t, r], axis=1)
        res = pd.concat(objs=[res, r], axis=0)
    res = res.reset_index(drop=True)
    toc = time.time()
    print('Time elapsed: {:.2f} seconds'.format(toc - tic))

    return res


def matches_to_phrase(matches, gap=' [.....] ', targetcol='target', colname='phrase', mark_left='', mark_right=''):
    """?"""

    matches = matches.copy()

    matches.left = matches.left.str.lower()
    matches.right = matches.right.str.lower()
    matches[targetcol] = mark_left + matches[targetcol].str.upper() + mark_right
    matches = matches.sort_values(by=['row', 'start'], ascending=[True, True])

    matches['start_next'] = matches.start.shift(-1)
    idx = matches.groupby('row')['start'].idxmax()
    matches.loc[idx, 'start_next'] = np.nan
    matches.start_next = matches.start_next.astype('Int64')

    matches['end_prev'] = matches.end.shift(1)
    idx = matches.groupby('row')['start'].idxmin()
    matches.loc[idx, 'end_prev'] = np.nan
    matches.end_prev = matches.end_prev.astype('Int64')

    matches['pad_left'] = matches.left.str.len()
    matches['pad_right'] = matches.right.str.len()

    # Update left side
    mask = matches.start - matches.end_prev <= matches.pad_left
    matches['left2'] = matches.left
    matches.loc[mask, 'left2'] = ''

    # Update right side
    mask = ~matches.start_next.isna()
    matches['right2'] = matches.right
    matches.loc[mask, 'right2'] = matches.loc[mask].apply(lambda x: x['right'][:x['start_next'] - 1 - x['end']], axis=1)

    # Gap
    mask = matches.start_next - matches.end > matches.pad_right
    matches['gap'] = ''
    matches.loc[mask, 'gap'] = gap

    # Phrases
    matches[colname] = matches['left2'] + matches[targetcol] + matches['right2']
    matches[colname] = matches[colname] + matches['gap']

    matches = matches.drop(labels=['start_next', 'end_prev', 'gap', 'pad_left', 'pad_right'], axis=1)

    # display(matches)
    r = matches.groupby('row')[colname].apply(''.join).reset_index()

    return r, matches
