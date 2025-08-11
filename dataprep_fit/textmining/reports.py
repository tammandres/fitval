from textmining.constants import VOCAB_DIR
from textmining.utils import extract, process, constrain_distance, check_vocab
from textmining.utils import get_context_patterns, remove_duplicates, pat_from_vocab, wrap_pat
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import regex as re
import time
from joblib import Parallel, delayed


VOCAB_FILE = 'vocab_site_and_tumour.csv'
CTX_FILE = 'context.csv'


def get_crc_reports(df: pd.DataFrame, col: str, vocab_path: Path = VOCAB_DIR, verbose: bool = False,
                    pad_left: int = 100, pad_right: int = 100, site_dist_left: int = 100,
                    site_dist_right: int = 100, site_dist_ex_left: int = 30, site_dist_ex_right: int = 30,
                    add_subj_to_matches: bool = False, subjcol: str = 'subject', spellcheck: bool = True,
                    negation_bugfix: bool = False):
    """Identify reports that describe current colorectal cancer

      df: Pandas DataFrame that contains a free-text report in each row
      col: name of column that contains reports
      verbose: if True, regex patterns that were used are printed out

    Output
      df : subset of df, where reports mention current colorectal cancer
      matches : shows all included and excluded matches for colorectal tumour keywords
    """
    tic = time.time()

    # ==== PATTERNS ====

    # Vocabulary for tumour site and tumour
    v = pd.read_csv(vocab_path / VOCAB_FILE)
    check_vocab(v)
    vsite = v.loc[v.cui.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
    print('\nSites included: {}'.format(vsite.concept.unique()))

    sites_ex = ['liver', 'lung', 'pelvis', 'uterus',
                'ovaries', 'bladder',
                'spleen', 'anastomosis', 'adrenal gland', 'kidney',
                'bone', 'pleura', 'brain', 'head', 'prostate']  # not mesentery, peritoneum, small intestine, nodes
    vsite_ex = v.loc[v.concept.isin(sites_ex)]
    print('Sites excluded: {}'.format(vsite_ex.concept.unique()))
    vtum = v.loc[v.cui == 11]
    vcrc = v.loc[v.cui == 12]

    # Vocabulary for context determination
    vcon = pd.read_csv(vocab_path / CTX_FILE)
    check_vocab(vcon)
    vcon = vcon.replace({'': np.nan})

    # Patterns for detecting CRC directly
    pats_crc = pat_from_vocab(vcrc, gap=r'\s{1,3}', verbose=verbose, name='CRC', spellcheck=spellcheck)  # gap=r'\s([\w\s]{0,10}\s)?'

    # For detecting tumours
    pats_tum = pat_from_vocab(vtum, gap=r'\s{1,3}', verbose=verbose, name='tumours', spellcheck=spellcheck)

    # For detecting sites
    # Sometimes site is not in same sentence, e.g. 'biopsy - colon. Moderate adenocarcinoma'. Hence char = '.'
    pats_site = pat_from_vocab(vsite, gap=r'\s{1,3}', verbose=verbose, name='sites', spellcheck=spellcheck)
    site_left = constrain_distance(pats_site, side='left', char='.', distance=site_dist_left)
    site_right = constrain_distance(pats_site, side='right', char='.', distance=site_dist_right)

    # For sites to exclude
    ex_site = pat_from_vocab(vsite_ex, gap=r'\s{1,3}', verbose=verbose, name='sites', spellcheck=spellcheck)
    site_ex_left = constrain_distance(ex_site, side='left', char='.', distance=site_dist_ex_left)
    site_ex_right = constrain_distance(ex_site, side='right', char='.', distance=site_dist_ex_right)

    # Additional exclusion patterns
    # pat_ex_left = wrap_pat(['metastatic', 'recurrent', 'recurring'])
    # pat_ex_left = constrain_as_word(pat_ex_left, pat_type='word', wdelim='\W')
    # pat_ex_left = constrain_distance(pat_ex_left, side='left', char=char, distance=20)

    # Patterns for detecting negation, generality, etc on left and right sides
    @dataclass
    class ContextPats:
        negated: str = None
        general: str = None
        historic: str = None
        possible: str = None
        metastatic: str = None
        response: str = None
        general2: str = None

    charset = r'[ \w\t\:\,\-\(\[\)\]\\\/]'
    gap = r'(?:\s{,3}[\w\(\)]+){,4}\s{,3}'  # r'([\w\(\)]+ ?){,4}\s{,3}'  # Allow up to 4 words with brackets, and make space optional
    pdist = None  # use pdist given in context.csv
    tdist = 100  # Termination pattern distance

    ctx_left = ContextPats()
    ctx_right = ContextPats()
    cats = ['negated', 'general', 'historic', 'possible', 'metastatic', 'response', 'general2']
    for c in cats:
        if verbose:
            print('\nGetting patterns for assigning tumour keywords to category: {}'.format(c))
        char = '.' if c == 'metastatic' else charset
        p_left = get_context_patterns(vcon, category=c, side='left', char=char, pdist=pdist, tdist=tdist, gap=gap,
                                      verbose=verbose)
        if negation_bugfix and c is'negated':
            gap_shorter = r'(?:\s{,3}[\w\(\)]+){,2}\s{,3}'
            vcon1 = pd.concat(objs=[vcon.iloc[[0]], vcon.loc[vcon.category.str.startswith('stop')]], axis=0)
            p_right1 = get_context_patterns(vcon1, category=c, side='right', char=char, pdist=pdist, tdist=tdist, gap=gap_shorter,
                                            verbose=verbose, output_list=True)
            
            vcon2 = vcon.drop(index=[0])
            p_right2 = get_context_patterns(vcon2, category=c, side='right', char=char, pdist=pdist, tdist=tdist, gap=gap,
                                            verbose=verbose, output_list=True)
            
            p_right = p_right1 + p_right2
            p_right = wrap_pat(p_right)
        else:
            p_right = get_context_patterns(vcon, category=c, side='right', char=char, pdist=pdist, tdist=tdist, gap=gap,
                                           verbose=verbose)
        ctx_left.__setattr__(c, p_left)
        ctx_right.__setattr__(c, p_right)

    # ==== PROCESS ====
    flags = re.IGNORECASE | re.DOTALL

    # ---- Get matches for colorectal tumour keywords, e.g. 'colorectal cancer' ----
    print('\nExtracting matches for colorectal tumour keywords...')
    crc = extract(df, col, pats_crc, flags=flags, pad_left=pad_left, pad_right=pad_right)
    crc['exclusion_indicator'] = 0
    crc['exclusion_reason'] = ''
    if verbose:
        print('Number of matches for colorectal tumour keywords: {}'.format(crc.shape[0]))

    # ---- Get matches for general tumour keywords, e.g. 'cancer' ----
    print('\nExtracting matches for general tumour keywords...')
    matches = extract(df, col, pats_tum, flags=flags, pad_left=pad_left, pad_right=pad_right)
    matches = matches.reset_index(drop=True)
    matches = remove_duplicates(crc, matches)
    matches['exclusion_indicator'] = 0
    matches['exclusion_reason'] = ''
    if verbose:
        print('Number of matches for general tumour keywords: {}'.format(matches.shape[0]))

    # ---- Mark general tumour keywords for exclusion, if not preceded or followed by colorectal site keyword ----
    if verbose:
        print('\nFinding tumour keywords not preceded or followed by colorectal tumour site...')

    # Get matches for tumour site
    print('\nExtracting matches for tumour site on left side...')
    sm_left = extract(matches, 'left', site_left, flags=flags) if site_left else pd.DataFrame()
    print('\nExtracting matches for tumour site on right side...')
    sm_right = extract(matches, 'right', site_right, flags=flags) if site_right else pd.DataFrame()
    print('\nExtracting matches for tumour site in the middle...')
    sm_middle = extract(matches, 'target', pats_site, flags=flags) if pats_site else pd.DataFrame()
    site_rows = pd.concat(objs=[sm_left, sm_right, sm_middle], axis=0).row

    # Mark for exclusion if no site
    matches['subrow'] = np.arange(matches.shape[0])
    matches.loc[~matches.subrow.isin(site_rows), 'exclusion_indicator'] = 1
    matches.loc[~matches.subrow.isin(site_rows), 'exclusion_reason'] += 'no site;'

    # Apply context patterns to sites detected on left and right sides, and mark for exclusion
    # This can be helpful, when site is more separated from main tumour keyword for subsequent context to work
    sm0 = pd.concat(objs=[sm_left, sm_right], axis=0).reset_index(drop=True)
    sm = sm0.copy()
    gen_left, hist_left = ctx_left.general, ctx_left.historic
    gen_right, hist_right = ctx_right.general, ctx_right.historic

    print('\nMarking sites for exclusion if site keywords historic or general..')
    sm = process(sm, 'left', [gen_left], action='remove', flags=flags) if gen_left else sm
    sm = process(sm, 'right', [gen_right], action='remove', flags=flags) if gen_right else sm
    sm = process(sm, 'left', [hist_left], action='remove', flags=flags) if hist_left else sm
    sm = process(sm, 'right', [hist_right], action='remove', flags=flags) if hist_right else sm

    sm1 = sm0.loc[np.setdiff1d(sm0.index, sm.index)]
    matches.loc[matches.subrow.isin(sm1.row), 'exclusion_indicator'] = 1
    matches.loc[matches.subrow.isin(sm1.row), 'exclusion_reason'] += 'site historic or general;'

    # ---- Mark general tumour keywords for exclusion, if they have unwanted site keyword nearby ----
    print('\nMarking general tumour keywords for exclusion if have unwanted site keyword nearby...')
    sm_ex_left = extract(matches, 'left', site_ex_left, flags=flags) if site_left else pd.DataFrame()
    sm_ex_right = extract(matches, 'right', site_ex_right, flags=flags) if site_right else pd.DataFrame()
    sm_ex = pd.concat(objs=[sm_ex_left, sm_ex_right], axis=0).reset_index(drop=True)
    matches.loc[matches.subrow.isin(sm_ex.row), 'exclusion_indicator'] = 1
    matches.loc[matches.subrow.isin(sm_ex.row), 'exclusion_reason'] += 'non-crc site;'

    if verbose:
        print('Number of matches without colorectal tumour site: {}'.format(matches.exclusion_indicator.sum()))

    # ---- Combine colorectal tumour keywords with general tumour keywords ----
    crc.index = np.arange(crc.shape[0]) + matches.index.max() + 1
    matches = pd.concat(objs=[matches, crc], axis=0).reset_index(drop=True)
    matches = matches.sort_values(by=['row', 'start'], ascending=[True, True])

    # ---- Identify tumour keywords that are negated, general, or historical ----
    # pats_left = [neg_left, gen_left, hist_left, pos_left, met_left]#, na_left]
    # pats_right = [neg_right, gen_right, hist_right, pos_right, met_right]#, na_right]
    # , 'not assessed']
    for cat in cats:
        print('\nExcluding tumour keywords in category ', cat)
        p_left = ctx_left.__getattribute__(cat)
        p_right = ctx_right.__getattribute__(cat)
        submatches = matches.copy()
        submatches = process(submatches, 'left', [p_left], action='remove', flags=flags) if p_left else submatches
        submatches = process(submatches, 'right', [p_right], action='remove', flags=flags) if p_right else submatches
        mask = ~matches.index.isin(submatches.index)
        matches.loc[mask, 'exclusion_indicator'] = 1
        matches.loc[mask, 'exclusion_reason'] += cat + ';'
        if verbose:
            print('Number of matches that are in {}: {}'.format(cat, mask.sum()))

    # Distinguish included and excluded tumour keywords
    # matches_excl = matches.loc[matches.exclusion_indicator == 1]  # Excluded matches
    matches_incl = matches.loc[matches.exclusion_indicator == 0]  # Included matches

    # Get reports that contain references to current coloretal tumours (tumour keywords that were included)
    df['row'] = np.arange(df.shape[0])
    dfsub = df.loc[df.row.isin(matches_incl.row)].copy()

    # Add subject back to matches
    if add_subj_to_matches:
        matches = matches.merge(df[['row', subjcol]], how='left', on='row')

    toc = time.time()
    print('Time elapsed: {} minutes'.format((toc - tic) / 60))
    return dfsub, matches


def get_targets(df: pd.DataFrame, col: str, target_pats: dict, vocab_path: Path = VOCAB_DIR, verbose: bool = False,
                pad_left: int = 100, pad_right: int = 100, site_dist_left: int = 100,
                site_dist_right: int = 100, site_dist_ex_left: int = 30, site_dist_ex_right: int = 30,
                add_subj_to_matches: bool = False, subjcol: str = 'patient_id', spellcheck: bool = True,
                negation_bugfix: bool = True):

    tic = time.time()
    # ---- 1. Prepare patterns ----

    # .... 1.1. Anatomical sites ....

    ## Vocab
    v = pd.read_csv(vocab_path / VOCAB_FILE)
    vsite = v.loc[v.cui.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
    print('\nSites included: {}'.format(vsite.concept.unique()))

    ## Excluded sites
    sites_ex = ['liver', 'lung', 'pelvis', 'uterus',
                'ovaries', 'bladder',
                'spleen', 'anastomosis', 'adrenal gland', 'kidney',
                'bone', 'pleura', 'brain', 'head', 'prostate']  # not mesentery, peritoneum, small intestine, nodes
    vsite_ex = v.loc[v.concept.isin(sites_ex)]
    print('Sites excluded: {}'.format(vsite_ex.concept.unique()))

    ## Vocabulary for context
    vcon = pd.read_csv(vocab_path / CTX_FILE)
    vcon = vcon.replace({'': np.nan})

    ## For detecting sites
    ## Sometimes site is not in same sentence, e.g. 'biopsy - colon. Moderate adenocarcinoma'. Hence char = '.'
    pats_site = pat_from_vocab(vsite, gap=r'\s{1,3}', verbose=verbose, name='sites', spellcheck=spellcheck)
    site_left = constrain_distance(pats_site, side='left', char='.', distance=site_dist_left)
    site_right = constrain_distance(pats_site, side='right', char='.', distance=site_dist_right)

    ## For sites to exclude
    ex_site = pat_from_vocab(vsite_ex, gap=r'\s{1,3}', verbose=verbose, name='sites', spellcheck=spellcheck)
    site_ex_left = constrain_distance(ex_site, side='left', char='.', distance=site_dist_ex_left)
    site_ex_right = constrain_distance(ex_site, side='right', char='.', distance=site_dist_ex_right)

    # .... 1.2. Patterns for detecting negation, generality, etc on left and right sides ....
    @dataclass
    class ContextPats:
        negated: str = None
        general: str = None
        historic: str = None
        possible: str = None
        metastatic: str = None
        response: str = None
        general2: str = None

    charset = r'[ \w\t\:\,\-\(\[\)\]\\\/]'
    gap = r'(?:\s{,3}[\w\(\)]+){,4}\s{,3}'  # r'([\w\(\)]+ ?){,4}\s{,3}'  # Allow up to 4 words with brackets, and make space optional
    pdist = None  # use pdist given in context.csv
    tdist = 100  # Termination pattern distance

    ctx_left = ContextPats()
    ctx_right = ContextPats()
    cats = ['negated', 'general', 'historic', 'possible', 'metastatic', 'response', 'general2']
    for c in cats:
        if verbose:
            print('\nGetting patterns for assigning tumour keywords to category: {}'.format(c))
        char = '.' if c == 'metastatic' else charset
        p_left = get_context_patterns(vcon, category=c, side='left', char=char, pdist=pdist, tdist=tdist, gap=gap,
                                        verbose=verbose)
        if negation_bugfix and c =='negated':
            gap_shorter = r'(?:\s{,3}[\w\(\)]+){,2}\s{,3}'
            vcon1 = pd.concat(objs=[vcon.iloc[[0]], vcon.loc[vcon.category.str.startswith('stop')]], axis=0)
            p_right1 = get_context_patterns(vcon1, category=c, side='right', char=char, pdist=pdist, tdist=tdist, gap=gap_shorter,
                                            verbose=verbose, output_list=True)
            
            vcon2 = vcon.drop(index=[0])
            p_right2 = get_context_patterns(vcon2, category=c, side='right', char=char, pdist=pdist, tdist=tdist, gap=gap,
                                            verbose=verbose, output_list=True)
            
            p_right = p_right1 + p_right2
            p_right = wrap_pat(p_right)
        else:
            p_right = get_context_patterns(vcon, category=c, side='right', char=char, pdist=pdist, tdist=tdist, gap=gap,
                                            verbose=verbose)
        ctx_left.__setattr__(c, p_left)
        ctx_right.__setattr__(c, p_right)


    # ---- 2. Process ----
    flags = re.IGNORECASE | re.DOTALL

    # .... 2.1. Get matches for targets ....
    print('\nExtracting matches for target keywords...')
    matches = pd.DataFrame()
    for cat, pat in target_pats.items():
        m = extract(df, col, pat, flags=flags, pad_left=pad_left, pad_right=pad_right)
        m['exclusion_indicator'] = 0
        m['exclusion_reason'] = ''
        m['category'] = cat
        matches = pd.concat(objs=[matches, m], axis=0)
        print('Number of matches for {} keywords {}'.format(cat, m.shape[0]))
    matches = matches.reset_index(drop=True)

    print('\nExtracted targets:')
    for cat in target_pats.keys():
        m = matches.loc[matches.category == cat]
        print(cat, ': ', m.target.str.lower().sort_values().unique())


    # .... 2.2. Mark target keywords for exclusion, if not preceded or followed by colorectal site keyword ----
    print('\nFinding target keywords not preceded or followed by colorectal tumour site...')

    # Get matches for tumour site
    print('\nExtracting matches for tumour site on left side...')
    sm_left = extract(matches, 'left', site_left, flags=flags) if site_left else pd.DataFrame()
    print('\nExtracting matches for tumour site on right side...')
    sm_right = extract(matches, 'right', site_right, flags=flags) if site_right else pd.DataFrame()
    print('\nExtracting matches for tumour site in the middle...')
    sm_middle = extract(matches, 'target', pats_site, flags=flags) if pats_site else pd.DataFrame()
    site_rows = pd.concat(objs=[sm_left, sm_right, sm_middle], axis=0).row

    # Mark for exclusion if no site
    matches['subrow'] = np.arange(matches.shape[0])
    matches.loc[~matches.subrow.isin(site_rows), 'exclusion_indicator'] = 1
    matches.loc[~matches.subrow.isin(site_rows), 'exclusion_reason'] += 'no site;'

    # Apply context patterns to sites detected on left and right sides, and mark for exclusion
    # This can be helpful, when site is more separated from main tumour keyword for subsequent context to work
    sm0 = pd.concat(objs=[sm_left, sm_right], axis=0).reset_index(drop=True)
    sm = sm0.copy()
    gen_left, hist_left = ctx_left.general, ctx_left.historic
    gen_right, hist_right = ctx_right.general, ctx_right.historic

    print('\nMarking sites for exclusion if site keywords historic or general..')
    sm = process(sm, 'left', [gen_left], action='remove', flags=flags) if gen_left else sm
    sm = process(sm, 'right', [gen_right], action='remove', flags=flags) if gen_right else sm
    sm = process(sm, 'left', [hist_left], action='remove', flags=flags) if hist_left else sm
    sm = process(sm, 'right', [hist_right], action='remove', flags=flags) if hist_right else sm

    sm1 = sm0.loc[np.setdiff1d(sm0.index, sm.index)]
    matches.loc[matches.subrow.isin(sm1.row), 'exclusion_indicator'] = 1
    matches.loc[matches.subrow.isin(sm1.row), 'exclusion_reason'] += 'site historic or general;'

    # .... Mark target keywords for exclusion, if they have unwanted site keyword nearby ....
    print('\nMarking target tumour keywords for exclusion if have unwanted site keyword nearby...')
    sm_ex_left = extract(matches, 'left', site_ex_left, flags=flags) if site_left else pd.DataFrame()
    sm_ex_right = extract(matches, 'right', site_ex_right, flags=flags) if site_right else pd.DataFrame()
    sm_ex = pd.concat(objs=[sm_ex_left, sm_ex_right], axis=0).reset_index(drop=True)
    matches.loc[matches.subrow.isin(sm_ex.row), 'exclusion_indicator'] = 1
    matches.loc[matches.subrow.isin(sm_ex.row), 'exclusion_reason'] += 'non-crc site;'
    print('Number of matches without colorectal tumour site: {}'.format(matches.exclusion_indicator.sum()))

    # .... Identify target keywords that are negated, general, or historical ....
    for cat in cats:
        print('\nExcluding tumour keywords in category ', cat)
        p_left = ctx_left.__getattribute__(cat)
        p_right = ctx_right.__getattribute__(cat)
        submatches = matches.copy()
        submatches = process(submatches, 'left', [p_left], action='remove', flags=flags) if p_left else submatches
        submatches = process(submatches, 'right', [p_right], action='remove', flags=flags) if p_right else submatches
        mask = ~matches.index.isin(submatches.index)
        matches.loc[mask, 'exclusion_indicator'] = 1
        matches.loc[mask, 'exclusion_reason'] += cat + ';'
        if verbose:
            print('Number of matches that are in {}: {}'.format(cat, mask.sum()))

    # .... Distinguish included and excluded keywords ....
    matches_incl = matches.loc[matches.exclusion_indicator == 0]  # Included matches

    # Get reports that contain included matches
    df['row'] = np.arange(df.shape[0])
    dfsub = df.loc[df.row.isin(matches_incl.row)].copy()

    # Add subject back to matches
    if add_subj_to_matches:
        matches = matches.merge(df[['row', subjcol]], how='left', on='row')

    toc = time.time()
    print('Time elapsed: {} minutes'.format((toc - tic) / 60))

    return dfsub, matches


def get_crc_reports_par(nchunks, njobs, df, col, **kwargs):
    
    test = df.shape[0] == df.index.nunique()
    if not test:
        raise ValueError("Index of df is not unique; run 'df.reset_index(drop=True)' first")
    
    def _iter(i):
        dfsub = df.loc[[i], [col]]
        __, m = get_crc_reports(df=dfsub, col=col, **kwargs)
        return m

    def _process_chunk(indices):
        out = pd.DataFrame()
        for i in indices:
            m = _iter(i)
            out = pd.concat(objs=[out, m], axis=0)
        return out

    if nchunks == 1:
        out = _process_chunk(df.index)
        out = [out]
    else:
        chunks = np.array_split(df.index, nchunks)
        out = Parallel(n_jobs=njobs)(delayed(_process_chunk)(indices) for indices in chunks)
    
    matches = pd.concat(objs=out, axis=0)

    return matches


def get_crc_reports_par(nchunks, njobs, df, col, **kwargs):
    tic = time.time()
    
    test = df.shape[0] == df.index.nunique()
    if not test:
        raise ValueError("Index of df is not unique; run 'df.reset_index(drop=True)' first")
    if nchunks < 2:
        raise ValueError("nchunks must be greater than 1 to run parallel processing of reports")
    
    indices = np.arange(df.shape[0])

    def _process_chunk(indices):
        dfsub = df.iloc[indices]
        __, m = get_crc_reports(df=dfsub, col=col, **kwargs)
        row_map = {i:row for i, row in enumerate(indices)}
        m['row'] = m.row.replace(row_map)
        return m
    
    chunks = np.array_split(indices, nchunks)
    out = Parallel(n_jobs=njobs)(delayed(_process_chunk)(indices) for indices in chunks)
    
    matches = pd.concat(objs=out, axis=0)
    matches = matches.sort_values(by=['row', 'start', 'end', 'target']).reset_index(drop=True)
    
    toc = time.time()
    print('Time elapsed: {} minutes'.format((toc - tic) / 60))

    return matches


def get_targets_par(nchunks, njobs, df, col, target_pats, **kwargs):
    tic = time.time()
    
    test = df.shape[0] == df.index.nunique()
    if not test:
        raise ValueError("Index of df is not unique; run 'df.reset_index(drop=True)' first")
    
    indices = np.arange(df.shape[0])

    def _process_chunk(indices):
        dfsub = df.iloc[indices]
        __, m = get_targets(df=dfsub, col=col, target_pats=target_pats, **kwargs)
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
