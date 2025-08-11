"""Patterns for extracting TNM values"""
from dataclasses import dataclass
from textmining.utils import wrap_pat, mix_case
import regex as re


# ==== Get information about TNM values ====
@dataclass
class TNM:
    """Stores information about TNM categories: their name, allowed letters, allowed values"""
    name: str
    letters: list
    values: list
    val_pat: str = None
    val_pat_roman: str = None


KEYS = ['T', 'N', 'M', 'L', 'V', 'R', 'SM', 'H', 'G', 'Pn']


def _tnm_info(key: str):
    """Returns information about a single TNM category, e.g. its name, letters, possible values.

      key: must be in ['pre', 'T', 'N', 'M', 'L', 'V', 'R', 'SM', 'H', 'Pn']

      Note that 'pre' returns a dataclass that contains possible values for prefixes
      NB. All letters in val_pat must appear in brackets, for subsequent case mixing function to work
    """

    pre = TNM(name='prefix',
              letters=[],
              values=['a', 'c', 'm', 'p', 'r', 'y', r'\(y\)'])

    t = TNM(name='T',
            letters=['T'],
            values=['0', '1', '1a', '1b', '1c', '1d', '2', '2a', '2b', '2c', '2d', '3', '3a', '3b', '3c', '3d',
                    '4', '4a', '4b', '4c', '4d', 'X', 'is'],
            val_pat=r'(?:[i][s]|[0X]|[1-4][abcd]?)')

    n = TNM(name='N',
            letters=['N'],
            values=['0', '1', '1a', '1b', '1c', '2', '2a', '2b', '2c', '3', '3a', '3b', '3c', 'X'],
            val_pat=r'(?:[0X]|[1-3][abc]?)')

    m = TNM(name='M',
            letters=['M'],
            values=['0', '1', '1a', '1b', '1c', 'X'],
            val_pat=r'(?:[0X]|1[abc]?)')

    lym = TNM(name='LymphaticInvasion',
              letters=['L', 'LV', 'Ly', 'L y'],
              values=['0', '1', 'X'],
              val_pat='[01X]')

    v = TNM(name='VenousInvasion',
            letters=['V'],
            values=['0', '1', '2', 'X'],
            val_pat='[012X]')

    r = TNM(name='ResidualTumourStatus',
            letters=['R'],
            values=['0', '1', '2', 'X'],
            val_pat='[012X]')

    sm = TNM(name='KikuchiLevel',
             letters=['SM', 'Sm', 'sm'],
             values=['1', '2', '3'],
             val_pat='[123]',
             val_pat_roman='[I]{1,3}')

    h = TNM(name='HaggittLevel',
            letters=['H'],
            values=['0', '1', '2', '3', '4'],
            val_pat='[01234]',
            val_pat_roman='[I][V]|[I]{1,3}')  # Important to have IV first, so it is matched before I

    pn = TNM(name='PerineuralInvasion',
             letters=['Pn', 'PNI'],
             values=['0', '1', 'X'],
             val_pat='[01X]')

    g = TNM(name='GradeOfDifferentiation',
            letters=['G'],
            values=['1', '2', '3', '4', 'X'],
            val_pat='[1234X]')

    tnm = dict(pre=pre, T=t, N=n, M=m, L=lym, V=v, R=r, SM=sm, H=h, Pn=pn, G=g)
    return tnm[key]


# ==== Simple TNM patterns ====
def _simple_tnm_value(single_pattern=True, capture=False, zero_and_is=True):
    """Constructs a simple pattern for matching TNM values such as 'T0', 'T1a', 'N1' etc.

        Captures repeated values ('T0/1/2').
        Requires each TNM value to be followed by a gap or by another TNM-like string, to protect from false positives.
        Is mainly used for constraining the context of more carefully constructed TNM value patterns.
        Has capture group for prefix, and for letter-value combination
    """
    # Possible prefixes with optional gap before
    pre = r'(?:([CPRYMcprym]{1,5}) ?)?' if capture else r'(?:[CPRYMcprym]{1,5} ?)?'

    # Patterns for individual TNM letters and values
    # These are less constrained and could lead to false positive matches more easily, especially as letter O is used
    if zero_and_is:
        pats = {'T': r'[Tt] ?(?:[O0-4Xx][A-Da-d]?|is)',
                'N': r'[Nn] ?[O0-3Xx][A-Ca-c]?',
                'M': r'[Mm] ?[O01Xx][A-Ca-c]?',
                'L': r'(?:[Ll]|[Ll][VvYy]|L y) ?[O01Xx]',
                'V': r'[Vv] ?[O012Xx]',
                'R': r'[Rr] ?[O012Xx]',
                'SM': r'[Ss][Mm] ?[123]',
                'H': r'[Hh] ?[0-4]',
                'G': r'[Gg] ?[1-4Xx]',
                'Pn': r'(?:Pn|PNI) ?[O01Xx]'
                }
    else:
        pats = {'T': r'[Tt] ?(?:[0-4Xx][A-Da-d]?)',
                'N': r'[Nn] ?[0-3Xx][A-Ca-c]?',
                'M': r'[Mm] ?[01Xx][A-Ca-c]?',
                'L': r'(?:[Ll]|[Ll][VvYy]|L y) ?[01Xx]',
                'V': r'[Vv] ?[012Xx]',
                'R': r'[Rr] ?[012Xx]',
                'SM': r'[Ss][Mm] ?[123]',
                'H': r'[Hh] ?[0-4]',
                'G': r'[Gg] ?[1-4Xx]',
                'Pn': r'(?:Pn|PNI) ?[01Xx]'
                }

    # Simple pattern for capturing repeated TNM values, such as 1/2/3 or 1a&b or T1/T2/T3
    rep = r'(?: {,2}[\-\,\/&] {,2}(?:(?:[tTnNmMlLvVrRhHgG]|sm|SM|Pn)?[O0-4Xx][A-Da-d]?|[A-Da-d])){,5}'

    # Tailing pattern to help avoid false positives like 'WELL TO MODERATELY', where TO MO could be mistaken for T0 M0
    # Ensures that the TNM value is followed by nonword character, or immediately by another TNM-like string
    # So the final pattern matches for TO in 'TOMO', 'TO MO', but not in 'TO MO[a-zA-Z]', as [a-zA-z] is not TNM like
    tail = r'(?=\b|[CPRYMcprym]{,5} ?[GTNMRLVgtnmrlv] ?[O0-4Xx])'
    front = r'(?<=\b|[CPRYMcprym]{,5} ?[GTNMRLVgtnmrlv] ?[O0-4Xx][a-dA-D]?)'

    if single_pattern:
        # A single pattern for capturing all TNM values
        let_val = wrap_pat(list(pats.values()))

        # Prefix + letter and value + repetition + tail
        let_val_rep = '(' + let_val + rep + ')' if capture else let_val + rep
        pat = '(?:' + front + pre + let_val_rep + tail + ')'
        return pat
    else:
        refined_pats = {}
        for key, let_val in pats.items():
            let_val_rep = '(' + let_val + rep + ')' if capture else let_val + rep
            refined_pats[key] = '(?:' + pre + let_val_rep + tail + ')'
        return refined_pats


def _simple_tnm_sequence(gapsize=100):
    """Constructs a pattern for extracting TNM sequences such as 'T1 N0 M0'
    Sequences must have at least 2 TNM values.
    Not used by default as can lead to more false positive matches,
    e.g. 'T1-weighted ... stage T2' would be picked up as a TNM sequence.
    """
    # Construct patterns for sequences of 2-4 TNM values
    ux = _simple_tnm_value(single_pattern=True, capture=False)
    gap = '[^:]{,' + str(gapsize) + '}'
    tnm_phrase = '(?:(?:' + ux + gap + '){1,3}' + ux + ')'
    return tnm_phrase


# ==== Constrained (more complex) TNM patterns ====
@dataclass
class ValuePattern:
    """Stores patterns for extracting TNM letters and values, for a single TNM category"""
    norm: str  # Normal pattern with no unusual variations, e.g. 'T0'

    case: str  # Allows letter to be lowercase, e.g. 't0'
    gap: str  # Allows gap after letter, e.g. 'T 0'
    mis: str  # Allows 0 to be mis-spelt as O, e.g. 'TO'

    norm_gap: str  # Matches for either .norm or .gap
    norm_gap_case: str  # Matches for either .norm, .case or .gap
    comb: str  # Matches for either .norm, .case, .gap, or .mis
    flex: str  # Pattern that allows for all variations at the same time, not just one variation as in case/gap/mis

    long: str = None  # Pattern where values have length 2, e.g. T1a, T1b, T1c.
    norm_pre: str = None  # Normal pattern that must have a prefix, e.g. 'pT0'


@dataclass
class ValueContainer:
    """Stores ValuePattern instances for each TNM category"""
    T: ValuePattern
    N: ValuePattern
    M: ValuePattern
    L: ValuePattern
    V: ValuePattern
    R: ValuePattern
    SM: ValuePattern
    H: ValuePattern
    G: ValuePattern
    Pn: ValuePattern


def _tnm_sequence(sequence_type: str = 'constrained', gapsize: int = 100, constrain_value_context: bool = True):
    """Combines patterns for individual TNM values to extract a sequence of TNM values.
    Extracting a sequence of TNM values, rather than individual values, helps to control for false positives.

    sequence_type:
      'constrained': match for specific sequences of TNM values, using constrained patterns for individual values
      'flexible': match for any sequence of TNM values, using constrained patterns for individual values
      'too_flexible': match for any sequence of TNM values, using non-constrained patterns for individual values
    """
    pats = _tnm_value(constrain_value_context)

    # Get combined pattern: normal, and variations in gap, case, 0=O
    tx, nx, mx, lx, vx, rx = pats.T.comb, pats.N.comb, pats.M.comb, pats.L.comb, pats.V.comb, pats.R.comb
    pnx, kx, hx, gx = pats.Pn.comb, pats.SM.comb, pats.H.comb, pats.G.comb
    ux = wrap_pat([tx, nx, mx, lx, vx, rx, pnx, kx, hx, gx])

    # Get combined pattern: normal pattern, and variations in gap and case
    t0, n0, m0 = pats.T.norm_gap_case, pats.N.norm_gap_case, pats.M.norm_gap_case
    l0, v0, r0 = pats.L.norm_gap_case, pats.V.norm_gap_case, pats.R.norm_gap_case

    # Get long pattern: values have length 2, such as 1a, 1b, 1c
    tlong = pats.T.long

    # Get normal pattern that always has prefix, e.g. pT1, ypT1
    tpre = pats.T.norm_pre

    # Construct patterns specific sequences of TNM values, having gaps in between
    #  gap is a longer gap, allowing any character within the gap except :
    #  gap2 is a short gap, allowing any character within the gap except :
    gap = '[^:]{,' + str(gapsize) + '}'
    gap2 = '[^:]{,10}'

    if sequence_type == 'constrained':

        # Require 3 letters: R..T..M/N/V/L, T..V..M/N/R/L, T..N..M/R/V/L
        # First part: permutations of TNMLVR, second part: optional tail of L-V-R-Pn-K-H-G
        a = rx + gap + tx + gap + wrap_pat([mx, nx, vx, lx]) + '(?:' + gap2 + wrap_pat([lx, vx, pnx, kx, hx, gx]) + ')?'
        b = tx + gap + vx + gap + wrap_pat([mx, nx, rx, lx]) + '(?:' + gap2 + wrap_pat([lx, rx, pnx, kx, hx, gx]) + ')?'
        c = tx + gap + nx + gap + wrap_pat([mx, rx, vx, lx]) + '(?:' + gap2 + wrap_pat([lx, vx, rx, pnx, kx, hx, gx]) + ')?'

        # T..N
        d = t0 + gap + n0  # Normal letters, gap and case allowed, e.g. T0 N0, T 0 N 0, t0 N 0
        e = t0 + gap2 + nx

        # R..T, T..V, T..M, T..R, N..M, N..V (do not allow lowercase)
        h = r0 + gap + t0
        i = t0 + gap + wrap_pat([v0, m0, r0])
        j = n0 + gap + wrap_pat([m0, v0])

        # T .. Kikuchi
        k = t0 + gap + kx

        # T..T, reduce false positives by requiring at least one T with modifier a/b/c/d, or both Ts with prefix
        l = t0 + gap + tlong + '(?:' + gap + ux + ')?'
        m = tlong + gap + t0 + '(?:' + gap + ux + ')?'
        n = tpre + gap + tpre + '(?:' + gap + ux + ')?'

        tnm_phrase = wrap_pat([a, b, c, d, e, h, i, j, k, l, m, n])  # f, g not incl

    elif sequence_type == 'flexible':

        # More flexible pattern, sequence of at least 2-4 TNM values
        tnm_phrase = '(?:(?:' + ux + gap + '){1,3}' + ux + ')'

    elif sequence_type == 'too_flexible':

        # A too flexible pattern for comparison
        tf, nf, mf, lf, vf, rf = pats.T.flex, pats.N.flex, pats.M.flex, pats.L.flex, pats.V.flex, pats.R.flex
        kf, hf, pnf, gf = pats.SM.flex, pats.H.flex, pats.Pn.flex, pats.G.flex
        uf = wrap_pat([tf, nf, mf, lf, vf, rf, kf, hf, pnf, gf])
        tnm_phrase = '(?:(?:' + uf + gap + '){0,3}' + ux + ')'

    return tnm_phrase


def _tnm_value(constrain_context: bool = True):
    """Returns a dataclass that contains multiple patterns for each TNM value"""
    pats = ValueContainer(T=_tnm_value_variations('T', constrain_context),
                          N=_tnm_value_variations('N', constrain_context),
                          M=_tnm_value_variations('M', constrain_context),
                          L=_tnm_value_variations('L', constrain_context),
                          V=_tnm_value_variations('V', constrain_context),
                          R=_tnm_value_variations('R', constrain_context),
                          SM=_tnm_value_variations('SM', constrain_context),
                          H=_tnm_value_variations('H', constrain_context),
                          G=_tnm_value_variations('G', constrain_context),
                          Pn=_tnm_value_variations('Pn', constrain_context)
                          )
    return pats


def _tnm_value_variations(key: str, constrain_context: bool = True):
    """Get patterns for extracting a single TNM category
        key: reference to TNM category, in ['T', 'N', 'M', 'L', 'V', 'R', 'SM', 'H', 'Pn']

    For example, if key='T', then
        .norm: pattern for typically written values, such as 'T0', 'T4a'

        .gap: pattern for TNM values with a gap after letter, such as 'T 0', 'T 4a'
        .case: pattern for TNM values where letter is lowercase, such as 't0', 't4a'
        .mis: pattern for TNM values where 0 is mis-spelt by O, such as 'TO'

        .comb: pattern that combines .norm, .gap, .case, and .mis patterns
        .flex: flexible pattern for matching all variations, such as 'T0', 'T 0', 't0', 'TO'

    Note
        The .comb category combines patterns that allow for one deviation of the normal pattern, e.g.
        either there is gap after the letter, or the letter is lowercase, but not both.
        The .flex category allows all deviations to occur at the same time and is more prone to false positives.
        Only patterns for T, N and M categories can have prefixes; for others, prefixes are not extracted.
    """
    info = _tnm_info(key)

    # Context for most patterns
    ctx = 'gap_or_value' if constrain_context else 'none'

    # Prefix
    if key in ['T', 'N', 'M']:
        prefix_values = _tnm_info('pre').values
        prefix = mix_case(wrap_pat(prefix_values))  # Letters in prefix can have mixed case
        prefix = prefix + '{1,5}'  # Prefix can be of length 1 to 5
    else:
        prefix = None

    # Letter
    letter_set = info.letters
    let_norm = wrap_pat(letter_set)
    let_low = let_norm.lower()
    let_up = let_norm.upper()
    let_mix = mix_case(let_norm)

    # Value
    value_set = info.values  # List of all possible values
    value = info.val_pat  # Regex for all possible values
    value = mix_case(value, add_bracket=False)

    # ==== Normal pattern: letters given in _tnm_info (usually uppercase), no gap, no mis-spelling ====
    #  For N category, create two versions to distinguish from perineural invasion:
    #  (1) prefix is either lowercase or optional
    #  (2) prefix is uppercase, and N value is preceded by T value at short distance (5 characters)
    if key == 'N':
        prefix_low = prefix.lower()  # Lowercase version of prefix
        norm1 = _build_tnm(pre=prefix_low, let=let_norm, gap='', val=value, rep=True, context=ctx)
        norm2 = _build_tnm(pre='P', let=let_norm, gap='', val=value, rep=True, context='preceded_by_t',
                           pre_optional=False)
        norm = wrap_pat([norm1, norm2])
    else:
        norm = _build_tnm(pre=prefix, let=let_norm, gap='', val=value, rep=True, context=ctx)

    # Add additional patterns for Kikuchi and Haggitt
    if key == 'H':
        value_with_roman = info.val_pat_roman
        value_with_roman = mix_case(value_with_roman, add_bracket=False)
        value_with_roman = wrap_pat([value_with_roman, value])

        h0 = mix_case(r'hagg?itt? {1,2}levels?(?:\W{0,3}is)?(?:\W{0,3}?at least)?[ \:\-\=]{1,2}(?:level\W{1,2})?H?')
        h1 = mix_case(r'hagg?itt?[ \:\-\=]{1,2}H?')
        letter_add = wrap_pat([h0, h1])
        norm_verbal = _build_tnm(pre=None, let=letter_add, gap=' {0,4}', val=value_with_roman, rep=True, context='gap')
        norm = wrap_pat([norm, norm_verbal])

    if key == 'SM':
        value_with_roman = info.val_pat_roman
        value_with_roman = mix_case(value_with_roman, add_bracket=False)
        value_with_roman = wrap_pat([value_with_roman, value])

        sm0 = mix_case(r'kikuchi {1,2}levels?(?:\W{0,3}levels?)?(?:\W{0,3}is)?(?:\W{0,3}?at least)?[ \:\-\=]{1,2}(?:level\W{1,2})?(?:SM)?')
        sm1 = mix_case(r'kikuchi[ \:\-\=]{1,2}(?:SM)?')
        sm2 = mix_case(r'SM level')
        letter_add = wrap_pat([sm0, sm1, sm2])
        norm_verbal = _build_tnm(pre=None, let=letter_add, gap=' {0,4}', val=value_with_roman, rep=True, context='gap')
        norm = wrap_pat([norm, norm_verbal])

    # ==== Gap after value ====
    gap = _build_tnm(pre=prefix, let=let_norm, gap=' ', val=value, rep=True, context=ctx)

    # ==== Variations in letter case ====
    #  when N is low (n), ensure prefix is low and there is not a preceding N value, o.w. cannot distinguish from Pn
    #  when Pn is low or up (pn or PN), ensure it is preceded by N, as otherwise cannot distinguish from N
    #  for kikuchi, allow mixed case, e.g. sm, SM, sM, Sm
    if key == 'N':
        prefix_low = prefix.lower()  # Lowercase version of prefix
        case = _build_tnm(pre=prefix_low, let=let_low, gap='', val=value, rep=True, context='not_preceded_by_n')
    elif key == 'Pn':
        case = _build_tnm(pre=prefix, let=wrap_pat([let_low, let_up]), gap='', val=value, rep=True, context='preceded_by_n')
    elif key == 'SM':
        case = _build_tnm(pre=prefix, let=let_mix, gap='', val=value, rep=True, context=ctx)
    else:
        case = _build_tnm(pre=prefix, let=let_low, gap='', val=value, rep=True, context=ctx)

    # ==== 0 mis-spelt as O ====
    if '0' in value_set:
        mis0 = _build_tnm(pre=prefix, let=let_norm, gap='', val='O', rep=False, context='value_before')
        mis1 = _build_tnm(pre=prefix, let=let_norm, gap='', val='O', rep=False, context='value_after')
        mis = wrap_pat([mis0, mis1])
        if key in ['T', 'N', 'M']:
            # Allow for things like pto, pno, pmo, pTo, pNo, pMo --> needs to have p prefix.
            mis2 = _build_tnm(pre='p', let=let_mix, gap='', val='o', rep=False, context='value_before', pre_optional=False)
            mis3 = _build_tnm(pre='p', let=let_mix, gap='', val='o', rep=False, context='value_after', pre_optional=False)
            mis = wrap_pat([mis, mis2, mis3])
    else:
        mis = None

    # ==== Normal pattern combined with patterns that have 1 deviation ====
    pats = [norm, gap, case, mis]
    comb = wrap_pat([p for p in pats if p is not None])
    norm_gap_case = wrap_pat([p for p in [norm, gap, case] if p is not None])
    norm_gap = wrap_pat([p for p in [norm, gap] if p is not None])

    # ==== Longer pattern: only include values that are longer than 1 char, e.g. '1a', '1b', 'is' for T category ====
    if key == 'T':
        val_long = mix_case(wrap_pat([v for v in value_set if len(v) > 1]))
        norm_long = _build_tnm(pre=prefix, let=let_norm, gap='', val=val_long, rep=True, context=ctx)
        gap_long = _build_tnm(pre=prefix, let=let_norm, gap=' ', val=val_long, rep=True, context=ctx)
        long = wrap_pat([norm_long, gap_long])
    else:
        long = None

    # ==== Normal pattern that forces a prefix ====
    if key in ['T', 'N', 'M']:
        norm_pre = _build_tnm(pre=prefix, let=let_norm, gap='', val=value, rep=True, context=ctx, pre_optional=False)
    else:
        norm_pre = None

    # ==== Flexible pattern ====
    # Needs to have just 3 capture groups - prefix, letter, value; don't use wrap_pat

    if '0' in value_set:
        value_all = wrap_pat([value, '[oO]'])
    else:
        value_all = value

    if key in ['SM', 'H']:
        # Include longer verbal patterns for kikuchi and Haggitt
        let_mix = wrap_pat([let_mix, letter_add])
        value_roman = info.val_pat_roman
        value_roman = mix_case(value_roman, add_bracket=False)
        value_all = wrap_pat([value_all, value_roman])
        flex = _build_tnm(pre=prefix, let=let_mix, gap=' ?', val=value_all, rep=True, context=ctx)
    else:
        flex = _build_tnm(pre=prefix, let=let_mix, gap=' ?', val=value_all, rep=True, context=ctx)

    # Gather
    pat = ValuePattern(comb=comb, norm=norm, gap=gap, case=case, mis=mis, flex=flex, long=long, norm_pre=norm_pre,
                       norm_gap_case=norm_gap_case, norm_gap=norm_gap)
    return pat


# ==== Build pattern for a single TNM value ====
def _build_tnm(pre: str, let: str, gap: str, val: str, rep=True, context='gap_or_value',
               pre_gap=' ?', pre_optional=True):
    """Construct a regex for a single TNM value by combining regexes for a prefix, a letter, and value"""

    # Prefix
    if pre is not None:
        pre = _prepare_prefix(pre, pre_gap)
        if pre_optional:
            pre = '(?:' + pre + ')?'
    else:
        pre = '()'  # Ensure empty capture group

    # Value
    if rep:
        val = _repeated_value(val, let=let)  # Value can be repeated, e.g. 'T1/2/3'
    val = '(' + val + ')'  # Value is captured

    # Letter
    let = '(' + let + ')'  # Letter is captured
    let = let + gap  # Letter is followed by gap, if specified

    # Add a little pattern before letter, to exclude certain combinations that could otherwise occur
    #  e.g. CYTO, CYNO, cyTO, CRT, crt, at, AT, AM, AN, art, arn etc
    ex = r'(?<![Cc][Yy](?=\w[Oo]))' + r'(?<![Cc][Rr](?=[Tt]))' + r'(?<!a(?=[tnm]))' + r'(?<![ACM])' + r'(?<![Aa][Rr])'
    let = ex + let

    # Basic pattern: prefix, letter, value
    pat = '(?:' + pre + let + val + ')'

    # Constrain context of the pattern
    pat = _constrain_context(pat, context=context)

    return pat


def _prepare_prefix(pre: str, pre_gap: str = ' ?'):
    """Prepares a prefix for TNM values (including the addition of capture group)"""

    # Add capture group
    pre = '(' + pre + ')'

    # Currently, I extract prefixes with a flexible pattern similar to r'[yprmca]{1,5}'
    # However, prefix should not be "ca ", "crc ", "a ", "c " (with gaps after)
    # Ways to handle this:
    # (1) Allow each letter only once in prefix -> BUT can lead to too many permutations
    # (2) Use 'logical OR and capture group method', i.e. 'ignore this pattern|(capture this pattern)'
    #     This is simple, but does not play well with subsequent code where I extract TNM phrases without capture groups
    # (3) So I resort to a more tedious method atm, by using lookbehinds and lookaheads at each position in string
    #ex = r'(?<!\b(?:[Cc][Aa]|[Cc][Rr][Cc]|[AaCc])\s)'
    ex0 = r'(?<!\bca\b)' + r'(?<!\bc(?=a\b))' + r'(?!\bca\b)'
    ex1 = r'(?<!\bcrc\b)' + r'(?<!\bcr(?=c\b))' + r'(?<!\bc(?=rc\b))' + r'(?!\bcrc\b)'
    ex2 = r'(?<!\b[ac]\s)' + r'(?<!\b[ac](?=\s))' + r'(?!\b[ac]\s)'
    ex = mix_case(ex0 + ex1) + mix_case(ex2, add_bracket=False)
    pre = pre + ex

    # Ignore subcategories of TNM values in prefix: e.g. 'c' is not part of prefix in T1cN0
    ex_sub = r'(?<![1-4](?=[a-dA-D]))'
    pre = ex_sub + pre

    # Prefix can have brackets around
    pre = r'[\(\[]?' + pre + r'[\)\]]?'
    pre = pre + pre_gap

    #pre = ex + '|' + ex_sub + '(' + pre + ')' + pre_gap
    return pre


def _repeated_value(val: str, let: str = None):
    """Expands a regex for TNM values, such that it captures repetitions for these values.
        val: regex for TNM values, such as r'[1-4]' or r'[1-4][a-d]?'
    For example, if val=r'[1-4][a-d]?', the updated pattern matches '3', '4/3', '4a/b', '4a & b'.
    However, it also matches for '1/c', which may or may not be valid.
    Also note that digit/digit patterns in brackets are not matched, e.g. '(2/3)' is not considered a repeated value
    as it is likely a lymph node count
    """

    # Values that can be repeated
    #  Include letter in the set of values that can be repeated, to capture things like T1/T2/T3, not just T1/2/3
    #  Adding [A-Da-d] is necessary as it is not matched by itself by TNM patterns
    if let is not None:
        pat = '(?:' + '(?:' + let + ')?' + val + '|[A-Da-d])'
    else:
        pat = '(?:' + val + '|[A-Da-d])'

    # Patterns for what can come before repeated value
    #  Patterns like '(0/2)', '(1/4)' etc are excluded as they may represent node counts rather than repeated values
    #  So if a pattern 'bracket - digit - slash - digit - bracket' occurs after value, only value is included
    p0 = r'[\-\,\/&]|[\(\[]? {,2}\bor\b'  # '-', ',', '/', '&', '( or', 'or'
    p1 = r'[\(\[](?!\s{,2}\d{1,3}\s{,2}\/\s{,2}\d{1,3}\s{,2}[\)\]])'   # Brackets not followed by digit/digit
    #if let is not None and re.search(let, 'nN'):  # Could make p1 work only when letter is N
    before = r'(?:' + p0 + '|' + p1 + ')'

    # Construct pattern for repeated value
    rep = '(?:' + val + r'(?: {,2}' + before + r' {,2}' + pat + '){,5}' + r'[\)\]]?' + ')'
    return rep


def _constrain_context(pat, context='gap_or_value'):
    """Constrain context of a TNM value
       pat: regex for TNM letter and value, such as r'T[1-4][a-d]?' or r'[acmpry]{1,5}T1[1-4]'
       context: specifies what the pattern must be preceded and followed by (see comments in code)

    For example, some false positives can be avoided by ensuring that each TNM value is followed by a gap
    or another TNM-like value: 'TO MO' could be a valid TNM string (0 mis-spelt as O), but 'TO MODERATELY' is not.

    In addition, defining the gap more explicitly helps avoid false positives such as T1/9,
    as '/' is not in charset for gap.
    """
    ctx = ['gap', 'gap_or_value', 'value_before', 'value_after', 'preceded_by_n', 'preceded_by_t',
           'not_preceded_by_n', 'none']
    if context not in ctx:
        raise ValueError('context must be in ' + str(ctx))

    tnm_pat = _simple_tnm_value(single_pattern=True, capture=False, zero_and_is=True)
    tnm_pat = r'\W?' + tnm_pat + r'\W?'  # To be able to capture pT1/pT2 for example - as space does not allow '/' atm
    space = r'[\s\:\;\,\[\]\(\)\.]'  # r'\W'

    if context == 'gap':
        # Preceded and followed by gap
        before = '(?<=^|' + space + ')'
        after = '(?=$|' + space + ')'
        return before + pat + after

    elif context == 'gap_or_value':
        # Preceded by gap or TNM-like value (or 'stage'), followed by gap or TNM-like value
        before = '(?<=^|' + space + '|' + tnm_pat + '|' + '[sS][tT][aA][gG][eE][dD]?' + ')'
        after = '(?=$|' + space + '|' + tnm_pat + ')'
        return before + pat + after

    elif context == 'value_before':
        # Preceded by a TNM-like value (or 'stage') (followed by a GAP or TNM-like value)
        before = '(?<=' + tnm_pat + r'\W{,3}' + '|' + '[sS][tT][aA][gG][eE][dD]?' + ')'
        after = '(?=$|' + space + '|' + tnm_pat + ')'
        return before + pat + after

    elif context == 'value_after':
        # Followed by a TNM-like value (preceded by a gap or TNM-like value)
        before = '(?<=^|' + space + '|' + tnm_pat + ')'
        after = r'(?=\W{,3}' + tnm_pat + ')'
        return before + pat + after

    elif context == 'preceded_by_n':
        # Preceded by N-like value, followed by gap or TNM-like value
        before = '(?<=[Nn][0-3Xx][ABCabc]?' + '[^:]{,10})'
        after = '(?=$|' + space + '|' + tnm_pat + ')'
        return before + pat + after

    elif context == 'not_preceded_by_n':
        # Not preceded by N-like value, followed by gap or TNM-like value
        before = '(?<![Nn][0-3Xx][ABCabc]?' + '[^:]{,10})'
        after = '(?=$|' + space + '|' + tnm_pat + ')'
        return before + pat + after

    elif context == 'preceded_by_t':
        # Preceded by T-like value in very short distance (5), followed by gap or TNM-like value
        before = '(?<=[Tt][0-4Xx][A-Da-d]?' + '[^:]{,5})'
        after = '(?=$|' + space + '|' + tnm_pat + ')'
        return before + pat + after

    elif context == 'none':
        return pat
