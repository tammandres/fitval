from textmining.constants import VOCAB_DIR
from pathlib import Path
from joblib import Parallel, delayed
from difflib import SequenceMatcher
from tqdm import tqdm
import pandas as pd
import numpy as np
import regex as re
import warnings


SPELL_PATH = Path(VOCAB_DIR / 'spellcheck.csv')

# ==== EXTRACT MATCHES, FILTER MATCHES, ROUGH SENTENCE BOUNDARIES FOR MATCHES ====
def extract(df: pd.DataFrame, col: str = 'safe_imaging_report', pat: str = 'pattern',
            pad_left: int = 100, pad_right: int = 100, flags=0, groups: bool = True):
    """Get matches for pattern 'pat' in column 'col' in dataframe 'df'
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
    for i, txt in enumerate(tqdm(reports)):
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
                        target = ['' if not m.group(i) else m.group(i) for i in range(n_group+1)]
                    else:
                        target = [target]
                        
                    data = [i, start, end, left] + target + [right]
                    s = pd.DataFrame(data).transpose()
                    s.columns = colnames
                    matches = pd.concat(objs=[matches, s], axis=0)
    
    # Tidy
    matches = matches.astype({'row': 'int32', 'start': 'int32', 'end': 'int32'})
    matches = matches.reset_index(drop=True).copy()
    return matches


def extract_set(df: pd.DataFrame, col: str = 'safe_imaging_report', pats: list = ['pattern1', 'pattern2'],
                pad_left: int = 40, pad_right: int = 40, flags=0, groups: bool = True):
    """Applies 'extract' function in a loop to extract matches for multiple patterns
    Output contains a 'pat_id' column that indicates which of the patterns the match corresponds to.
    """
    matches = pd.DataFrame()
    for i, pat in enumerate(pats):
        m = extract(df, col, pat=pat, pad_left=pad_left, pad_right=pad_right, flags=flags, groups=groups)
        m['pat_id'] = i
        matches = pd.concat(objs=[matches, m], axis=0)
    matches = matches.reset_index(drop=True)  # Ensure index is unique
    return matches


def process(df, col, pats=['pattern0', 'pattern1'], action='keep', flags=re.IGNORECASE | re.DOTALL):
    """Remove or keep rows in df, depending on whether 'col' matches patterns in 'pats'"""
    if pats:
        test = df.index.nunique() != df.shape[0]
        if test:
            warnings.warn("Index of input DataFrame contains duplicate values. This can lead to unexpected results")

        # Extract matches
        m = extract_set(df, col, pats=pats, pad_left=100, pad_right=100, flags=flags)

        # Remove or keep rows in the dataframe 'df'
        if action == 'remove':
            idx_drop = df.iloc[m.row.drop_duplicates(), :].index
            df = df.drop(idx_drop, axis=0)
        elif action == 'keep':
            idx_keep = df.iloc[m.row.drop_duplicates(), :].index
            df = df.loc[idx_keep, :]
    else:
        warnings.warn("Argument 'pats' is empty, returning the original input DataFrame")
    return df


def remove_duplicates(d0, d1):
    """Remove matches from d1, that are already contained in d0

        d0: DataFrame with columns 'row', 'start', 'end', 'target' - produced by textmining.utils.extract
        d1: DataFrame with columns 'row', 'start', 'end', 'target' - produced by textmining.utils.extract
    """
    t0 = d0[['row', 'start', 'end']].rename(columns={'start': 'start0', 'end': 'end0'}).copy().reset_index(drop=True)
    t1 = d1[['row', 'start', 'end']].copy().reset_index(drop=True)

    t1 = t1.merge(t0, how='left')
    t1['row_mask'] = (t1['start'] >= t1['start0']) & (t1['end'] <= t1['end0'])
    t1 = t1.groupby(['row', 'start', 'end'])['row_mask'].max().reset_index()
    t1 = t1[~t1.row_mask].drop_duplicates()

    d1 = d1.merge(t1, how='left')
    d1 = d1.loc[d1.row_mask == False].drop(labels='row_mask', axis=1)

    return d1


def get_sentence(matches, flags=0, s=r'[^\n\r\t\.\?;]'):
    """Get rough sentence boundaries using characters in set 's'
    Input: output of extract() or extract_set()
    Output: adds 'sentence_left', 'sentence_right' and 'sentence' to matches
    """
    matches = matches.copy()
    rows = pd.DataFrame(np.arange(matches.shape[0]), columns=['row'])

    # Get left part of the sentence, add to matches
    m = extract(matches, 'left', s+'{1,}$', flags=flags)
    m = m[['row', 'target']].rename(columns={'target':'sentence_left'})
    m = rows.merge(m, how='left', on='row').fillna('').drop('row',axis=1)  # When match was not found
    matches['sentence_left'] = m['sentence_left'].values
    
    # Get right part of the sentence, add to matches
    m = extract(matches, 'right', '^'+s+'{1,}')
    m = m[['row', 'target']].rename(columns={'target':'sentence_right'})
    m = rows.merge(m, how='left', on='row').fillna('').drop('row',axis=1)  # When match was not found
    matches['sentence_right'] = m['sentence_right'].values
    
    # Get sentence
    matches['sentence'] = matches['sentence_left'] + matches['target'] + matches['sentence_right']
    return matches


# ==== CHANGE CASE OF REGEX PATTERNS ====
# This allows some parts of regex to have a mixed case, and other parts to have an upper or lower case.
def revert_case(string):
    """Useful for examining patterns that had mix_case() applied to it"""
    s = re.sub(r"\[\w(\w)\]", r"\g<1>", string, flags=re.I)
    return s


def _case_repl(match):
    """Helper function for _mix_case()"""
    return '[' + match.group().lower() + match.group().upper() + ']'


def _case_repl_nb(match):
    """Helper function for _mix_case()"""
    return match.group().lower() + match.group().upper()


def mix_case(string, add_bracket=True):
    """Converts a string to regex where each letter is in two possible cases, ignoring escaped characters.
        if add_bracket is True: 'abC' -> '[aA][bB][cC]'
        if add_brackets is False: 'abC' -> 'aAbBcC'
    """
    if add_bracket:
        return re.sub(r'(?<!\\)[a-zA-Z]', _case_repl, string)
    else:
        return re.sub(r'(?<!\\)[a-zA-Z]', _case_repl_nb, string)


# ==== CONSTRUCT, CONSTRAIN, AND EXPAND REGEX PATTERNS ====
def wrap_pat(pats, sort=True, join_str='|', nocapture=True):
    """Converts a list of strings into a regular expression
    If sort=True, sorts strings in reverse order by length
    For example, pats=['b', 'cat', 'a'], sort=True, and join_str='|' gives '(?:cat|a|b)'
    """
    if sort:
        pats = sorted(pats, key=len, reverse=True)
    pat = '(?:' + join_str.join(pats) + ')' if nocapture else '(' + join_str.join(pats) + ')'
    return pat


def _not_preceded_by(string: str):
    """Input must contain letters in [a-zA-Z] only"""
    pat = r'(?<!\b' + mix_case(string) + ')'
    for i in range(1, len(string)):
        pat += r'(?<!\b' + mix_case(string[:i]) + '(?=' + mix_case(string[i:]) + '))'
    return pat


PAT_TYPES = ['word', 'wordstart', 'wordend', 'string']


def constrain_as_word(pat: str, pat_type: str = 'word', wdelim: str = r'\W'):
    """Constrain a string to match for words
        pat: string to be constrained
        pat_type:
            'word': string needs to match for a word, hence add word delimiters on both sides
            'wordstart': string is the start of a word, add word delimiter on left side, expand right with word char
            'wordend': string is the end of a word, add word delimiter on right side, expand left with word char
            'string': string is any part of a word, expand left and right sides with word characters
        wdelim: character that delimits word boundaries
    """
    if pat_type not in PAT_TYPES:
        raise ValueError("pat_type must be in {}".format(PAT_TYPES))

    if pat_type == 'word':
        start = r'(?<=' + wdelim + r'|^)'
        end = r'(?=' + wdelim + r'|$)'
    elif pat_type == 'wordstart':
        start = r'(?<=' + wdelim + r'|^)'
        end = r'\w*'
    elif pat_type == 'wordend':
        start = r'\w*'
        end = r'(?=' + wdelim + r'|$)'
    elif pat_type == 'string':
        start = r'\w*'
        end = r'\w*'
    pat = start + '(?:' + pat + ')' + end  # add non-capture group for patterns such as a|b|c
    return pat


def constrain_distance(pat: str, side: str = 'right', char: str = r'.', distance: int = 40):
    """Constrain regular expression to look for matches at a certain distance from the left or right side"""
    if distance is None:  # Any distance
        dist = '*'
    elif distance > 0:   # Specified distance
        dist = '{,' + str(distance) + '}'
    elif distance == 0:  # No distance
        dist = '{0}'

    if side == 'left':
        pat = pat + '(?=' + char + dist + '$|$)'
    elif side == 'right':
        pat = '(?<=^' + char + dist + '|^)' + pat

    return pat


def constrain(pats: list, side: str = 'left', pat_type: str = 'wordstart', char: str = '.',
              distance: int = 40, wdelim: str = r'\W'):
    """Applies constrain_as_word and constrain_distance to a list of patterns.

      pats: list of regular expresssions
      pat_type: must be in ['word', 'wordstart', 'wordend', 'string']
      char: regular expression for characters that are allowed between the expression, and start or end of string
      distance: distance to the start or end of string
      wdelim: character that delineates word boundaries
    """
    # Constrain as word
    pats = [constrain_as_word(p, pat_type=pat_type, wdelim=wdelim) for p in pats]

    # Constrain distance
    pats = [constrain_distance(p, side=side, char=char, distance=distance) for p in pats]

    return pats


def constrain_and_remove(df, pats, side, pat_type, char, distance, test=False, wdelim=r'\W'):
    """Constrains patterns in pats, the removes rows from df that match for patterns.
    df is an output of extract() or extract_set(), so it contains columns 'left', 'right' that are used for filtering.
    """
    if side == 'both':
        pats_left = constrain(pats, side='left', pat_type=pat_type, char=char, distance=distance, wdelim=wdelim)
        df = process(df, 'left', pats_left, action='remove', flags=re.DOTALL | re.IGNORECASE)

        pats_right = constrain(pats, side='right', pat_type=pat_type, char=char, distance=distance, wdelim=wdelim)
        df = process(df, 'right', pats_right, action='remove', flags=re.DOTALL | re.IGNORECASE)

    elif side == 'target':
        pats_target = constrain(pats, side='none', pat_type=pat_type, char=char, distance=distance, wdelim=wdelim)
        df = process(df, 'target', pats_target, action='remove', flags=re.DOTALL | re.IGNORECASE)

    elif side in ['left', 'right']:
        pats = constrain(pats, side=side, pat_type=pat_type, char=char, distance=distance, wdelim=wdelim)
        df = process(df, side, pats, action='remove', flags=re.DOTALL | re.IGNORECASE)
        if test:
            print('-------\n{}'.format(pats))
    return df


def prep_pat(pat: str, pat_type: str, gap: str = r'\s{1,3}', wdelim: str = r'\W', nocapture: bool = True):
    """Converts a string to a more elaborate regex:
        * expands or constrains it to match for words
        * extends any gaps (' ') it contains: gaps will be replaced by a regex given by the gap argument
        * removes capture groups, and optionally adds a new capture group for the new pattern

    Args
        pat: string to be converted
        pat_type: type of string, in ['word', 'wordstart', 'wordend', 'string']
        gap: regex that will replace all gaps within the string
        wdelim: character that delimits word boundaries
        nocapture: if False, final pattern will be wrapped in a capture group
    """
    # Make spaces flexible   #([\w\(\)]+ ){,5}([\w\(\)]+){,1}
    pat = re.sub(r'\s{1,}', gap, pat)

    # Remove capture group if exists (do this iteratively, in case there are nested groups)
    for __ in range(5):
        pat = re.sub(r'\((?!\?)(.*)?\)', r'(?:\g<1>)', pat)  #re.sub('\(', '(?:', pat)

    # Constrain as word or wordstart
    pat = constrain_as_word(pat=pat, pat_type=pat_type, wdelim=wdelim)

    # Wrap in a non-capture group, or a capture group
    pat = '(?:' + pat + ')' if nocapture else '(' + pat + ')'

    return pat


# ==== PREPARE PATTERNS BASED ON VOCABULARY GIVEN IN A DATAFRAME ====
def check_vocab(v: pd.DataFrame):
    """Check vocabulary"""
    test = v.pat_type.isin(PAT_TYPES)
    if not all(test):
        r = v.pat_type[~test].unique()
        raise ValueError("pat_type contains values in {}, but it must only contain values in {}".format(r, PAT_TYPES))


def prep_excl_pats(cats: list, rules: pd.DataFrame, gap: str = r'\s{1,3}', wdelim: str = r'\W'):
    """Prepare exclusion patterns for context determination
        cats: list of categories
        rules: Pandas DataFrame that contains rules
    Note: partially duplicates pat_from_vocab() function --> merge the two?
    """
    ex = []
    for cat in cats:
        rules_ex = rules.loc[rules.category == cat]
        pats_ex = [prep_pat(pat=r.pat, pat_type=r.pat_type, gap=gap, wdelim=wdelim) for i, r in rules_ex.iterrows()]
        [ex.append(p) for p in pats_ex]
    ex = wrap_pat(ex)
    return ex


def pat_from_vocab(vocab: pd.DataFrame, gap: str = r'\s{1,3}', verbose: bool = True, name: str = None,
                   spellcheck: bool = False):
    """Construct patterns for detecting a category specified in vocab"""
    if (name is not None) and verbose:
        print('\nGetting patterns for detecting {} ...'.format(name))

    pats = []
    for i, row in vocab.iterrows():
        pat = prep_pat(pat=row.pat, pat_type=row.pat_type, gap=gap, wdelim=r'\W')
        if verbose:
            print(pat)
        pats.append(pat)
    pats = wrap_pat(pats)

    if spellcheck:
        spell = pd.read_csv(SPELL_PATH)
        pats = add_spell_variation(pats, spell)

    return pats


def get_context_patterns(rules: pd.DataFrame, category: str, side: str, gap: str = r'\s{1,3}', char: str =r'[ \w\t\:]',
                         pdist: int = None, tdist: int = 100, verbose: bool = False, output_list: bool = False):
    """Construct regex patterns for context determination,
    based on a DataFrame that contains rules in a specific format.

        rules: DataFrame that contains rules
        category: category of the concept to be detected, e.g. 'negation'

    char is set to r'[ \w\t\:]' as otherwise picks up matches such as "involvement of margins (R2): No\rTumour ..."
    tdist controls distance for terminating the rule
    """
    rsub = rules.loc[(rules.category == category) & (rules.side.isin([side, 'both']))]
    if rsub.shape[0] > 0:
        pats = []
        for i, row in rsub.iterrows():
            pat = prep_pat(pat=row.pat, pat_type=row.pat_type, gap=gap)
            if not pd.isnull(row.ex_left):
                cats = row.ex_left.split('|')
                tpat = prep_excl_pats(cats, rules, gap)
                pre = r'(?<!' + tpat + '.{,' + str(tdist) + '}' + ')'  # Not preceded by exclusion patterns
                pat = pre + pat
            if not pd.isnull(row.ex_right):
                cats = row.ex_right.split('|')
                tpat = prep_excl_pats(cats, rules, gap)
                post = r'(?!' + '.{,' + str(tdist) + '}' + tpat + ')'  # Not followed by exclusion patterns
                pat = pat + post
            if pdist is None:
                if not pd.isna(row.distance):
                    dist = int(row.distance)
                else:
                    dist = None
            else:
                dist = pdist
            pat = constrain_distance(pat, side=side, char=char, distance=dist)
            pats.append(pat)
            if verbose:
                print(pat)
        if not output_list:
            pats = wrap_pat(pats)
    else:
        pats = None
    return pats


def get_minmax(m, col_sort, col_variable, col_value, suf_min='_min', suf_max='',
               indecision=True, indecision_only=False):
    """Gets minimum and maximum values of extracted variables per report

    Args
     m : Pandas DataFrame with columns
         row : report number
         col_variable : names of variables extracted from the report (e.g. pT, pN)
         col_value : values of the extracted variables (e.g. 1, 2)
         col_sort : column that will be used to sort values (e.g. if col_value is string, col_sort can be a numeric representation of it)
         suf_min : suffix to add to minimum value of the variable, e.g. if variable is 'pT' and suffix is '_min', there will be 'pT_min'
         suf_max : suffix to add to maximum value of the variable
    """
    # Sort by report ('row'), variable, and (sortable) value
    m = m.sort_values(['row', col_variable, col_sort], ascending=False)
    variables = m[col_variable].unique()

    # Get maximum values
    smax = m[['row', col_variable, col_value]].groupby(['row', col_variable], as_index=False).first()
    pmax = smax.pivot(index='row', columns=col_variable, values=col_value)
    cols = {c: c + suf_max for c in pmax.columns if np.isin(c, variables)}
    pmax = pmax.rename(columns=cols)
    smax = smax.rename(columns={col_value: col_value + suf_max})

    # Get minimum values
    smin = m[['row', col_variable, col_value]].groupby(['row', col_variable], as_index=False).last()
    pmin = smin.pivot(index='row', columns=col_variable, values=col_value)
    cols = {c: c + suf_min for c in pmin.columns if np.isin(c, variables)}
    pmin = pmin.rename(columns=cols)
    smin = smin.rename(columns={col_value: col_value + suf_min})

    # Combine
    p = pmax.merge(pmin, how='outer', left_index=True, right_index=True)

    # Combine minimum and maximum values for each report to create an indecision flag
    if indecision:
        sind = pd.merge(smin, smax, how='inner', on=['row', col_variable])
        sind['indecision'] = 0
        sind.loc[sind[col_value + suf_min] != sind[col_value + suf_max], 'indecision'] = 1
        pind = sind.pivot(index='row', columns=col_variable, values='indecision')
        pind.columns = pind.columns + '_indecision'
        if indecision_only:
            p = pind.copy()
        else:
            p = p.merge(pind, how='outer', left_index=True, right_index=True)
    return p


def add_spell_variation(pat: str, spell: pd.DataFrame):
    """Expands pattern in pat, so it matches for spelling variations of pat

    spell is a DataFrame with columns
        'repl': contains word for which spelling variations are sought, e.g. 'tumour'
        'pat': contains spelling variations of the word, e.g. 'tmr|tumor|tymour'

    Assumes pat contains sub-patterns separated by empty space ' ', e.g. 'sigmoid colon'
    Each sub-part is separately replaced by its spelling variations, e.g. '(?:sigmoid|sgmoid) (?:clon|colon)'
    """
    newpat = []
    subpats = pat.split(' ')
    for s in subpats:
        for __, row in spell.iterrows():
            match = re.search(s, row.repl)
            if match:
                pat_add = r'\b(?:' + row.pat + r')\b'
                s = wrap_pat([s, pat_add], sort=False)
        newpat.append(s)
    return ' '.join(newpat)


# ==== ASSESS OVERLAP BETWEEN TWO SETS OF CLINICAL REPORTS ====
def get_matching_blocks(r1, r2, max_size=10, nchunks=20, njobs=-1):
    """Assess partial overlap between two sets of clinical reports
    Args
        r1: reports in set 1, a list
        r2: reports in set 2, a list
        max_size: maximum size of overlapping text to consider
        nchunks: reports in set 1 are divided into nchunks number of chunks to be processed in parallel
        njobs: number of parallel processes
    Returns
        matches: a DataFrame of overlapping text for all report pairs
        sizes: a list of total sizes of overlapping strings across all reports
            e.g. if report 1 and report 2 have three overlaps, with sizes 10, 15 and 14, the total size is 39
    """
    if not (isinstance(r1, list) and isinstance(r2, list)):
        raise ValueError('inputs r1 and r2 must be python lists')

    def _process_chunk(indices):  # Processes reports from set 1 that match indices, and all reports from set 2
        sizes = []
        matches = pd.DataFrame()
        for i in indices:
            report1 = r1[i]  # A single report from set 1 (string)
            for j in range(len(r2)):
                report2 = r2[j]  # A single report from set 2 (string)
                matcher = SequenceMatcher(None, report1, report2)
                matching_blocks = matcher.get_matching_blocks()  # Get all matching blocks
                matching_blocks = [b for b in matching_blocks if b.size >= max_size]  # Only include matches of size max_size
                if matching_blocks:
                    print('report', i, 'and report', j, 'have', len(matching_blocks), 'match(es)', end='\r')
                    substrings = [report1[match.a:match.a + match.size] for match in matching_blocks]  # All matching substrings
                    size = sum(match.size for match in matching_blocks)  # Total size of the overlap between report1 and report2
                    m = pd.DataFrame({'i': i, 
                                      'j': j, 
                                      'r1': report1,
                                      'r2': report2, 
                                      'overlaps': '[...]'.join(substrings)}, index=[(i, j)]
                                      )
                    matches = pd.concat(objs=[matches, m], axis=0)
                    #matches += m
                    sizes.append(size)
        matches['length'] = matches.overlaps.apply(len)
        return matches, sizes
    
    indices = range(len(r1))  
    chunks = np.array_split(indices, nchunks)  # Divide reports from set1 into nhcunks chunks
    out = Parallel(n_jobs=njobs)(delayed(_process_chunk)(indices) for indices in chunks)  # Process chunks in parallel

    sizes = []
    matches = pd.DataFrame()
    for output in out:
        matches = pd.concat(objs=[matches, output[0]], axis=0)
        sizes += output[1]
    matches = matches.sort_values(by=['length'], ascending=False)

    return matches, sizes
