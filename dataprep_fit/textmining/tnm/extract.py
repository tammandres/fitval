"""Functions for extracting TNM phrases: single TNM values, and TNM sequences"""
from textmining.constants import VOCAB_DIR
from textmining.utils import extract, process, get_sentence, wrap_pat, constrain_and_remove, remove_duplicates
from textmining.tnm.pattern import _tnm_value, _tnm_sequence, _simple_tnm_sequence, _simple_tnm_value
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import regex as re
import time
import warnings


# Default patterns for single TNM values and TNM sequences
PAT_SINGLE = _tnm_value(constrain_context=True)
PAT_SINGLE_SIMPLE = _simple_tnm_value()
PAT_TNM = _tnm_sequence(sequence_type='constrained')
PAT_TNM_FLEX = _tnm_sequence(sequence_type='flexible')


RULE_PATH = VOCAB_DIR / 'rules_tnm.csv'


def tnm_phrase(df: pd.DataFrame, col='report', gapsize: int = 100, pad_left: int = 200, pad_right: int = 200,
               extract_solitary: bool = True, flex_start: bool = False, simplicity: int = 2):
    """Extract all phrases that contain TNM values: sequences (e.g. 'T1 N0 M1') or single values (e.g. 'stage: T1')
    Some phrases are marked for exclusion ('exclusion_indicator' column equals 1), but not removed.

    Input
      df: Pandas dataframe
      col: column in the dataframe that contains report texts
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

    Output
     matches: DataFrame with columns ...
        'start', 'end': start and end positions of each match
        'target': matches for TNM phrases
        'exclusion_indicator': indicates whether target was marked for exclusion
        'exclusion_reason': reason for marking a target for exclusion
    """
    if simplicity not in [0, 1, 2]:
        raise ValueError('simplicity must be 0, 1, or 2')
    tic = time.time()

    # Extract TNM sequences
    print('\nExtracting TNM sequences ...')

    if simplicity == 2:
        pat_tnm = _tnm_sequence(sequence_type='constrained', gapsize=gapsize, constrain_value_context=True)
        pat_flex = _tnm_sequence(sequence_type='flexible', gapsize=gapsize, constrain_value_context=True)
    elif simplicity == 1:
        pat_tnm = _tnm_sequence(sequence_type='flexible', gapsize=gapsize, constrain_value_context=True)
        pat_flex = None
        if flex_start:
            warnings.warn("As simplicity is 1, setting flex_start to False")
            flex_start = False
    elif simplicity == 0:
        pat_tnm = _simple_tnm_sequence(gapsize=gapsize)
        pat_flex = None
        if flex_start:
            warnings.warn("As simplicity is 0, setting flex_start to False")
            flex_start = False

    matches = _extract_tnm_sequence(df, col, pat_tnm=pat_tnm, pat_flex=pat_flex, pad_left=pad_left, pad_right=pad_left,
                                    flex_start=flex_start)
    matches['solitary_indicator'] = 0

    # Extract matches for individual TNM values that are not part of a sequence
    if extract_solitary:
        print('\nExtracting single TNM values ...')

        if simplicity in [1, 2]:
            pats = _tnm_value(constrain_context=True)
            m = _extract_individual_tnm(df, col, pats=pats, pad_left=pad_left, pad_right=pad_right)
        elif simplicity == 0:
            pats = _simple_tnm_value(single_pattern=False, capture=False, zero_and_is=False)
            m = _extract_individual_tnm_simple(df, col, pats=pats, pad_left=pad_left, pad_right=pad_right)

        m = remove_duplicates(matches, m)  # Remove single TNM values if already contained within found sequences
        m['solitary_indicator'] = 1

        matches = pd.concat(objs=[matches, m], axis=0)
        matches = matches.reset_index(drop=True)  # Because concat can lead to non-unique index

    # Get approximate sentence involving matches (add 'sentence_left', 'sentence_right' and 'sentence')
    matches = get_sentence(matches)

    # Rank matches based on position in the pathology report (earlier matches have smaller ranks)
    ranks = matches.groupby(['row'], as_index=False)['start'].rank().rename(columns={'start': 'rank'})
    matches = matches.merge(ranks, how='left', left_index=True, right_index=True)

    toc = time.time()
    print('Time elapsed: {:.2f} minutes'.format((toc - tic) / 60))
    return matches


def _extract_tnm_sequence(df: pd.DataFrame, col: str, pat_tnm: str = PAT_TNM, pat_flex: str = PAT_TNM_FLEX,
                          flex_start: bool = False, pad_left: int = 200, pad_right: int = 200):
    """Extracts a sequence of TNM values, such as "T0 N0 M0"

        df: Pandas Dataframe
        col: name of column that contains reports
        pat_tnm: regex that matches TNM sequences
        pat_flex: optional regex that matches TNM sequences more flexibly than pat_tnm
        flex_start: if True, the flexible regex (pat_flex) is first used, and all matches that are also not captured
                    by the standard regex (pat_tnm) are marked for removal, but not removed.
                    This helps to examine whether anything may have been missed due to using a constrained pattern.
        pad_left, pad_right: number of characters to extract from the left and right of each match,
                             to help inspect the context of matches
    """
    flags = re.DOTALL

    if flex_start:
        # First find matches using a flexible pattern
        matches = extract(df, col, pat=pat_flex, pad_left=pad_left, pad_right=pad_right, flags=flags, groups=False)
        matches = matches[['row', 'start', 'end', 'left', 'target', 'right']]
        matches['exclusion_indicator'] = 0  # Add exclusion indicator for later
        matches['exclusion_reason'] = ''

        # Mark matches for removal if they do not fit a more constrained TNM sequence
        matches = get_sentence(matches)
        submatches = process(matches, 'sentence', pats=[pat_tnm], action='keep', flags=re.DOTALL)
        mask = ~matches.index.isin(submatches.index)
        matches.loc[mask, 'exclusion_indicator'] = 1
        matches.loc[mask, 'exclusion_reason'] += 'not constrained sequence;'
    else:
        matches = extract(df, col, pat=pat_tnm, pad_left=pad_left, pad_right=pad_right, flags=flags, groups=False)
        matches = matches[['row', 'start', 'end', 'left', 'target', 'right']]
        matches['exclusion_indicator'] = 0  # Add exclusion indicator for later
        matches['exclusion_reason'] = ''

    print('{} matches for TNM sequences'.format(matches.shape[0]))
    return matches


def _extract_individual_tnm(df: pd.DataFrame, col: str, pats=PAT_SINGLE, pad_left: int = 200, pad_right: int = 200,
                            include_flex: bool = True, rule_path: Path = RULE_PATH):
    """Extract matches for TNM values that are not part of a sequence, such as "Stage: T0".

    By default, normal patterns, and variations in gap and case are used.
    Patterns where 0 is mis-spelt as O are included, but marked for exclusion.
    Extracted patterns are further marked for exclusion if they contain certain patterns to the left and right.
    """
    flags = re.DOTALL

    # Patterns for TNM values
    pats_dict = asdict(pats)
    keys = list(pats_dict.keys())

    # Inclusion and exclusion patterns
    r = pd.read_csv(rule_path)
    all_keep = r.loc[(r.tnm_category == 'all') & (r.action == 'keep')]
    all_rm = r.loc[(r.tnm_category == 'all') & (r.action == 'remove')]

    # Extract
    all_matches = pd.DataFrame()
    for key in keys:
        print('\nExtracting individual TNM values for category: {}'.format(key))

        # Find matches for individual TNM values, using norm, gap, and case patterns
        pats_all = pats_dict[key]
        pats_use = [pats_all['norm'], pats_all['gap'], pats_all['case']]
        pats_use = [p for p in pats_use if p is not None]
        pat = wrap_pat(pats_use)
        matches = extract(df, col, pat, pad_left=pad_left, pad_right=pad_right, flags=flags, groups=False)
        matches['exclusion_indicator'] = 0
        matches['exclusion_reason'] = ''

        # Apply inclusion and exclusion rules to matches
        mask_cat = r.tnm_category.str.findall(r'\b' + key + r'\b').apply(len) != 0

        keep = r.loc[mask_cat & (r.action == 'keep')]
        keep = pd.concat(objs=[keep, all_keep], axis=0)

        rm = r.loc[mask_cat & (r.action == 'remove')]
        rm = pd.concat(objs=[rm, all_rm], axis=0)

        matches = _apply_rules(matches, keep, rm)

        # Find additional matches, using norm_pre patterns
        pats_add = [pats_all['norm_pre']]
        pats_add = [p for p in pats_add if p is not None]
        if pats_add:
            pat = wrap_pat(pats_add)
            m = extract(df, col, pat, pad_left=pad_left, pad_right=pad_right, flags=flags, groups=False)
            m['exclusion_indicator'] = 0
            m['exclusion_reason'] = ''

            # Only apply exclusion rules -- not inclusion rules,
            # so that things like 'perforation (pT4): No' are excluded, but pT4 is included (if no more excl words near)
            m = _apply_rules(m, keep=pd.DataFrame(), rm=rm)

            # Note: norm_pre patterns are a special case of norm pattern
            # Remove previously found TNM values if they were also found here, as otherwise there are duplicates
            matches = remove_duplicates(m, matches)
            matches = pd.concat(objs=[matches, m], axis=0).reset_index(drop=True)

        # Find additional matches where 0 mis-spelt as O -- these are marked for exclusion
        if include_flex:
            pats_add = [pats_all['mis']]
            pats_add = [p for p in pats_add if p is not None]
            if pats_add:
                pat = wrap_pat(pats_add)
                m = extract(df, col, pat, pad_left=pad_left, pad_right=pad_right, groups=False)
                m['exclusion_indicator'] = 1
                m['exclusion_reason'] = 'single TNM value with flexible pattern;'
                matches = pd.concat(objs=[matches, m], axis=0).reset_index(drop=True)

        all_matches = pd.concat(objs=[all_matches, matches], axis=0)
        print('{} matches, {} marked for exclusion'.format(matches.shape[0], matches.exclusion_indicator.sum()))
    return all_matches


def _extract_individual_tnm_simple(df: pd.DataFrame, col: str, pats=PAT_SINGLE_SIMPLE,
                                   pad_left: int = 200, pad_right: int = 200):
    """Extract matches for TNM values that are not part of a sequence, such as "Stage: T0".
    Using the simplest of TNM patterns.
    """
    flags = re.DOTALL

    # Patterns for TNM values
    pats_dict = pats
    keys = list(pats_dict.keys())

    # Inclusion and exclusion patterns
    r = pd.read_csv(RULE_PATH)
    all_keep = r.loc[(r.tnm_category == 'all') & (r.action == 'keep')]
    all_rm = r.loc[(r.tnm_category == 'all') & (r.action == 'remove')]

    # Extract
    all_matches = pd.DataFrame()
    for key in keys:
        print('\nExtracting individual TNM values for category: {}'.format(key))

        # Find matches for individual TNM values, using norm, gap, and case patterns
        matches = extract(df, col, pats_dict[key], pad_left=pad_left, pad_right=pad_right, flags=flags)
        matches['exclusion_indicator'] = 0
        matches['exclusion_reason'] = ''

        # Apply inclusion and exclusion rules to matches
        mask_cat = r.tnm_category.str.findall(r'\b' + key + r'\b').apply(len) != 0

        keep = r.loc[mask_cat & (r.action == 'keep')]
        keep = pd.concat(objs=[keep, all_keep], axis=0)

        rm = r.loc[mask_cat & (r.action == 'remove')]
        rm = pd.concat(objs=[rm, all_rm], axis=0)

        matches = _apply_rules(matches, keep, rm)

        all_matches = pd.concat(objs=[all_matches, matches], axis=0)
        print('{} matches, {} marked for exclusion'.format(matches.shape[0], matches.exclusion_indicator.sum()))
    return all_matches


def _apply_rules(matches: pd.DataFrame, keep: pd.DataFrame, rm: pd.DataFrame):
    """Filter matches based on inclusino and exclusion words"""

    # Mark reports for exclusion if they have no inclusion words nearby
    wdelim = r'\b'  # r'[ \t\(\[\)\]\-\*\:]' -> without \b, things like "perforation (pT4): no\r" may be missed
    if not keep.empty:
        char = r'[^\n\r]'
        submatches = matches.copy()
        for i, row in keep.iterrows():
            submatches = constrain_and_remove(submatches, pats=[row.string], side=row.side, pat_type=row.string_type,
                                              char=char, distance=row.distance, wdelim=wdelim)
        mask = matches.index.isin(submatches.index)
        matches.loc[mask, 'exclusion_indicator'] = 1
        matches.loc[mask, 'exclusion_reason'] += 'single TNM value with no inclusion words;'

    # Mark reports for exclusion if they have exclusion words nearby
    # NB - if wdelim does not contain :, then things like 'perforation:no' cannot be detected.
    wdelim = r'\b'  # r'[ \t\(\[\)\]\-\*\:]'
    if not rm.empty:
        char = r'[^\n\r\.]'  # More flexible, e.g. "Stage T1. No budding" -- no is not referring to T1
        submatches = matches.copy()
        for i, row in rm.iterrows():
            submatches = constrain_and_remove(submatches, pats=[row.string], side=row.side, pat_type=row.string_type,
                                              char=char, distance=row.distance, wdelim=wdelim)
        mask = ~matches.index.isin(submatches.index)
        matches.loc[mask, 'exclusion_indicator'] = 1
        matches.loc[mask, 'exclusion_reason'] += 'single TNM value with exclusion words;'

    return matches
