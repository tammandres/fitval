import numpy as np
import pandas as pd
import regex as re
import time
import warnings
from textmining.utils import constrain, extract_set, get_sentence, process, get_minmax


def get_pn(df: pd.DataFrame, col: str, varname: str = 'Pn'):
    """Extracts perineural invasion where it is given in words (e.g. 'perineural invasion: yes')
    rather than in TNM notation (e.g. Pn1)
    """
    tic = time.time()
    print('Extracting additional perineural invasion')
    cols = np.array(['Pn', 'Pn_min', 'Pn_indecision'])
    test = np.isin(cols, df.columns)
    if test.any():
        warnings.warn(
            'Some columns of input data have the same name as columns that will be added: these will first be removed')
        df = df.drop(cols[test], axis=1)

    # Patterns
    concept = ['perineural invasion']

    right1 = ['yes', 'present',
              'is (|[^,]{0,10} )identified',
              'is (|[^,]{0,10} )seen',
              'is (|[^,]{0,10} )noted',
              'is (|[^,]{0,10} )mural',
              'are  (|[^,]{0,10} )seen']
    conj = ['but', 'although', 'though']
    right0 = ['no', 'none', 'absent',
              'not (|[^,]{0,10} )identified',
              'not (|[^,]{0,10} )seen',
              'not (|[^,]{0,10} )noted',
              'not (|[^,]{0,10} )mural']
    right1 = constrain(right1, 'right', pat_type='word', distance=10, char='[^\,]')
    right0 = constrain(right0, 'right', pat_type='word', distance=10, char='[^\,]')

    left1 = ['there {1,3}is', 'there {1,3}are', 'possible', 'extensive']
    left0 = ['no', 'no {1,3}evidence', 'not']
    left1 = constrain(left1, 'left', pat_type='word', distance=50, char='[^\,]')
    left0 = constrain(left0, 'left', pat_type='word', distance=50, char='[^\,]')

    # Extract all matches for perineural invasion
    matches = extract_set(df, col, concept, flags=re.IGNORECASE | re.DOTALL, pad_left=200, pad_right=200)
    matches = get_sentence(matches)
    matches['variable'] = varname
    matches['value'] = ''

    # Proceed if any matches found
    if matches.shape[0] > 0:

        # Matches where Pn is not present
        m = process(matches, 'sentence_right', right0, action='keep')  # Retain matches that match for right0
        m = process(m, 'sentence_left', left1, action='remove')  # Remove if match for left1 on the left
        m = process(m, 'sentence_right', conj, action='remove')  # Remove if match for conjunctions on the right
        m2 = process(matches, 'sentence_left', left0, action='keep')  # Retain if match for left0
        m = pd.concat(objs=[m, m2], axis=0).drop_duplicates()  # Combine to get a list of negative matches
        matches.loc[matches.index.isin(m.index), 'value'] = '0'

        # Get matches where Pn is present
        mx = matches.loc[~matches.index.isin(m.index)]
        p = process(mx, 'sentence_right', right1, action='keep')  # Retain matches that match for right1
        p2 = process(mx, 'sentence_left', left1, action='keep')
        p = pd.concat(objs=[p, p2], axis=0).drop_duplicates()
        matches.loc[matches.index.isin(p.index), 'value'] = '1'

        # Get minimum and maximum values
        m = matches.loc[matches.value != ''].copy()
        m['value_tmp'] = m.value.astype(float)
        r = get_minmax(m, col_sort='value_tmp', col_variable='variable', col_value='value', suf_min='_min', suf_max='')
        r = r.reset_index()

        # Add to reports
        df['row'] = np.arange(df.shape[0])
        df = df.merge(r, how='left', on='row')
        df = df.drop('row', axis=1)
    else:
        for c in cols:
            df[c] = np.nan

    toc = time.time()
    print('Time elapsed: {:.2f} seconds'.format(toc - tic))
    return df, matches
