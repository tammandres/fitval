""" Extract clinical symptoms relevant for colorectal cancer

Almost all symptom keywords were taken from previous symptom extraction pipeline by Diana Withrow
Some keywords for detecting negation and for applying a ConText-like algorithm
were taken from https://github.com/chapmanbe/pyConTextNLP/tree/master/KB
by Wendy W. Chapman, Glenn Dayton, Danielle Mowery
"""
from textmining.constants import VOCAB_DIR
from textmining.spelling import update_pat
from pathlib import Path
import textmining.utils as ut
import numpy as np
import pandas as pd
import regex as re
import time
import warnings

SPELL_PATH = Path(VOCAB_DIR / 'spellcheck.csv')
SYM_OUT = 'sym.csv'


# nicesym = dfsym.loc[dfsym.category.isin(syms)]
# nsym = fit.loc[fit.subject.isin(nicesym.subject)].subject.nunique()
# ntot = fit.subject.nunique()
# print('\n{} ({:.2f}%) of {} patients have records for NICE symptoms'.format(nsym, nsym/ntot*100, ntot))


def symptoms(run_path: Path, df, col: str = 'clind_all', spell_path: Path = SPELL_PATH, save_data: bool = False):
    """Get symptoms from GP clinical details
        df: clinical details DataFrame for patients with FIT values, outputted by fitml.dataprep.fit_and_clind
        run_path: where to save results
        spell_path: path to spell check table
        save_data: if True, data saved to SYM_OUT file
    """
    if 'subject' not in df.columns:
        raise ValueError("df must have a column 'subject' for each individual")

    # Spell check lookup
    spell = pd.read_csv(spell_path)

    # Patterns
    pat = symptom_pats(spell, trim_spell=True, reorder_groups=True)
    print(pat.category.unique())

    # Get matches
    matches, __ = extract_symptoms(df, dfpat=pat, col=col, categories=None, remove_marked=False)
    print(matches.category.unique())

    # Get symptoms for each subject
    matches_incl = matches.loc[matches.exclusion_indicator == 0]
    dfsym = matches_incl[['row', 'category']].drop_duplicates() #add_symptoms(df, matches_incl, form='long')
    df['row'] = np.arange(df.shape[0])
    dfsym = df[['subject', 'row']].merge(dfsym, how='inner')

    # Explore initial counts
    s = dfsym.groupby('category')['subject'].nunique()
    print(s.sort_values(ascending=False))

    # Compute the proportion of patients that have any FIT-relevant symptoms
    print('\nAll symptoms: {}'.format(dfsym.category.unique()))
    syms = ['abdopain', 'abdomass', 'ida', 'anaemia', 'tarry', 'bloodsympt', 'diarr',
            'bowelhabit', 'constipation', 'rectalpain', 'rectalmass', 'rectalulcer', 'wl']
    print('\nSymptoms in NICE: {}'.format(syms))

    if save_data:
        raise NotImplementedError
        #df.to_csv(run_path / SYM_OUT, index=False)

    return dfsym, matches


def explore_symptom_matches(matches: pd.DataFrame):

    # Excluded matches
    ex = matches.loc[matches.exclusion_indicator == 1]

    # Unique values for included matches
    s = matches.loc[matches.exclusion_indicator == 0]
    s = s.groupby(['category', 'target', 'len_target']).size().rename('n_match').reset_index()
    s = s.sort_values(by=['category', 'n_match', 'len_target'], ascending=[True, False, False])
    s = s[['category', 'target', 'n_match']]

    return ex, s


def symptom_pats(spell: pd.DataFrame, trim_spell: bool = True, reorder_groups: bool = False):
    """Get regex for extracting symptoms
        spell: DataFrame for spelling patterns, corresponds to 'spellcheck.csv'
        trim_spell: if True, spelling patterns are shortened for faster and more general result

    NB: gaps are written as ( ) -> captured, to detect certain exclusion words in gaps.
    """
    dfpat = pd.DataFrame()

    # Words inside a pattern can be separated by such a gap that can contain other words
    gap = r'[ \t\:\\\/]{1,3}(?:\w{1,10}[ \t\:\\\/])?'
    gap_short = r'[ \t\:\\\/]{1,3}(?:\w{1,4}[ \t\:\\\/])?'

    # Words that must not occur in gaps inside patterns
    #  Don't include 'if' in set1 as can be misspelling of 'in', e.g. 'change if bowel habit'
    #  Don't include 'for', 'from'
    #  'screening' is included, as in 'bowel screening increased'
    set1 = r'\b(?:screening|although|apart|aside|but|except|however|nevertheless|still|though|yet|which)\b'
    set2 = r'\b(?:no|nil|not|none|without|cannot|absent|excluded|free|clear|negative|resolved|ruled out)\b'
    set3 = r'\b(?:and)'
    gap_ex = set1 + '|' + set2 
    gap_ex_three = set1 + '|' + set2 + '|' + set3

    # ==== Abdominal pain ====

    # Abdominal pain
    pat0 = r'(?:abdomen|abdominal|bowels|bowel|belly|abdo|lif|rif)( )(?:ache|aches|aching|cramp|cramps|cramping|crampy|discomfort|pain|pains)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['abdopain', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # Pain ... abdomen
    pat0 = r'(?:ache|aches|aching|cramp|cramps|cramping|crampy|discomfort|pain|pains)( )(?:abdomen|abdominal|bowels|bowel|belly|abdo|lif|rif)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['abdopain', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Abdominal mass ====

    # Abdominal mass
    pat0 = r'(?:abdomen|abdominal|bowels|bowel|belly|abdo|lif|rif)( )(?:mass|lump)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['abdomass', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # mass ... abdomen
    pat0 = r'(?:mass|lump)( )(?:abdomen|abdominal|bowels|bowel|belly|abdo|lif|rif)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['abdomass', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Rectal mass ====

    # rectal mass
    pat0 = r'(?:anal|anus|rectum|rectal|perineum|perineal|perine\w*)( )(?:mass|lump)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['rectalmass', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # mass ... rectum
    pat0 = r'(?:mass|lump)( )(?:anal|anus|rectum|rectal|perineum|perineal|perine\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['rectalmass', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Low iron, and iron deficiency anaemia ====

    # low ... iron
    pat0 = r'(?:low|dip|dips|drop|drops|dropped|dropping|fall|falls|fallen|falling)( )(?:fe|iron)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['low_iron', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # iron ... low
    pat0 = r'(?:fe|iron)( )(?:low|dip|dips|drop|drops|dropped|dropping|fall|falls|fallen|falling)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['low_iron', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # iron ... deficiency (but no anaemia mentioned)
    pat0 = r'(?:fe|iron)( )(?:deficiency|deficient|def)(?!.{,20}(?:anaemia|anemia|anemic|anaemic|anaem\w*|anem\w*))'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['low_iron', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # iron ... deficiency ... without ... anaemia
    pat0 = r'(?:fe|iron)( )(?:deficiency|deficient|def)( )(?:without)( )(?:anaemia|anemia|anemic|anaemic|anaem\w*|anem\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['low_iron', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # iron ... deficiency ... anemia
    # Important that gap is inside an optional capture group as otherwise unwanted words picked up (AT: perhaps not anymore as not optional)
    pat0 = r'(?:fe|iron)( )(?:deficiency|deficient|def)(?:( )(?:anaemia|anemia|anemic|anaemic|anaem\w*|anem\w*))'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['ida', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ida
    pat0 = r'(?:ida)'
    pat = update_pat(pat0, spell, gap=None, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['ida', pat0, pat, None]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Anaemia. To distinguish from IDA with more certainty, exclude previously detected 'ida' category later ====

    # low ... haemoglobin
    pat0 = r'(?:low|dip|dips|drop|drops|dropped|dropping|fall|falls|fallen|falling)( )(?:haemoglobin|hemoglobin|hb|haemog\w*|hemog\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['anaemia', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # haemoglobin ... low
    pat0 = r'(?:haemoglobin|hemoglobin|hb|haemog\w*|hemog\w*)( )(?:low|dip|dips|drop|drops|dropped|dropping|fall|falls|fallen|falling)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['anaemia', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # anaemia
    pat0 = r'(?<!(?:fe|iron)( )(?:deficiency|deficient|def)( ))(?:anaemia|anemia|anemic|anaemic|anaem\w*|anem\w*|microcytosis|microcy\w*)'
    #pat0 = r'(?:anaemia|anemia|anemic|anaemic|anaem\w*|anem\w*|microcytosis|microcy\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['anaemia', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)
    #re.search(pat, 'iron def anemia')
    #m=extract(df, 'clind' pat, flags=re.I|re.DOTALL)  # use pat with lookbehind
    #m2=extract(df, 'clind' pat, flags=re.I|re.DOTALL)  # use pat without lookbehind
    #m2.loc[~m2.row.isin(m.row)].left.drop_duplicates()  # difference: iron def

    # Hb in range 20-99 g/L, Hb in range 100-119. g/L, Hb in range 10-11.9 g/dL, Hb in range 0-9.9 g/dL
    #  have ( )? for cases like Hb70 -- when there's no gap
    pat0 = r'(?:hemoglobin|hb|haemog\w*|hemog\w*)( )?\b(?:[2-9][0-9]|1[01][0-9]|1[01]|[0-9])\b'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['anaemia', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Tarry ~ 'not fresh blood in stool' ====

    # tarry, darker stool
    pat0 = r'tarry|(?:melena|melaena)|(?:black|dark|darker|tarry)( )(?:stool|stools|fecal|faecal|faeces|feces)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['tarry', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # occult blood
    pat0 = r'(?:occ|occult)( )(?:blood|bloods|bloody|bleed|bled|bleeding|bleedings|bleeds)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['tarry', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Blood symptoms ~ 'fresh blood in stool'; excludes tarry and occult blood ====

    # blood in stool
    pat0 = r'(?<!(?:occ|occult).{,10})(?:blood|bloods|bloody|bleed|bled|bleeding|bleedings|bleeds|red|red stain\w*|red colo\w*)( )(?:stool|stools|fecal|faecal|faeces|feces)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bloodsympt', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # blood ... bowel
    pat0 = r'(?:blood|bloods|bloody|bleed|bled|bleeding|bleedings|bleeds)( )(?:bowel|bowels)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bloodsympt', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # bowel ... blood
    pat0 = r'(?:bowel|bowels)( )(?:blood|bloods|bloody|bleed|bled|bleeding|bleedings|bleeds)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bloodsympt', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # rectal blood, haemorrhage
    pat0 = r'(?:anal|anus|rectum|rectal|perineum|perineal|perine\w*)( )(?:blood|bloods|bloody|bleed|bled|bleeding|bleedings|bleeds|haemorrhage|hemorrhage|hemor\w*|haemor\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bloodsympt', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    ## -> Also include it unders separate rectal bleeding category
    tmp = pd.DataFrame([['rectalbleed', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # bleeding, incl pr bleed, bleed pr; but excluding menstrual blood, blood in urine
    pat0 = r'(?:(?:pr|interm\w*)( ))?(?<!(?:hosp\w*|hospital|menst\w*|menstrual|menstr\w*).{,10})(?:blood|bloods|bloody|bleed|bled|bleeding|bleedings|bleeds)(?!.{,10}(?:urine|urin\w*).{,10})(?:( )(?:pr|interm\w*))?'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bloodsympt', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # additional patterns for separate rectal bleeding category 
    pat0 = r'(?:(?:pr)( ))(?<!(?:hosp\w*|hospital|menst\w*|menstrual|menstr\w*).{,10})(?:blood|bloods|bloody|bleed|bled|bleeding|bleedings|bleeds)(?!.{,10}(?:urine|urin\w*).{,10})'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['rectalbleed', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    pat0 = r'(?<!(?:hosp\w*|hospital|menst\w*|menstrual|menstr\w*).{,10})(?:blood|bloods|bloody|bleed|bled|bleeding|bleedings|bleeds)(?!.{,10}(?:urine|urin\w*).{,10})(?:( )(?:pr))'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['rectalbleed', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # print(re.sub(pat, '<detected>', 'blood in urine'))
    # print(re.sub(pat, '<detected>', 'blood'))

    # ==== Diarrhoea ====

    # diarrhoea
    pat0 = r'(?:diarr|diarrh|diarrhoea|diarrhea)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['diarr', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # loose ... stool
    pat0 = r'(?:loose|looser|runny|liquid|liquid\w*|watery|urgency)( )(?:stool|stools|motion|motions|bowel|bowels|bo)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['diarr', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # stool ... loose
    pat0 = r'(?:stool|stools|motion|motions|bowel|bowels|bo)( )(?:loose|looser|runny|liquid|liquid\w*|watery|urgency)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['diarr', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Constipation ====

    # constipation
    pat0 = r'(?:constip|constipat|constipation|constipated)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['constipation', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # difficulty passing stool
    pat0 = r'(?:difficult|difficulty)( )(?:pass|passing)( )(?:stool|stools|fecal|faecal|faeces|feces)'
    pat = update_pat(pat0, spell, gap_short, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['constipation', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # faecal loading
    pat0 = r'(?:stool|stools|fecal|faecal|faeces|feces)( )(loading)'
    pat = update_pat(pat0, spell, gap_short, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['constipation', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Change in bowel habit, includes constipation and diarrhoea --> this is added later ====

    # Shorthand
    pat0 = r'(?:cbh|abh|cibh|cib|cob|ci bh|chinbh|ch bh)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bowelhabit', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # Individual patterns for bowel, habit, change
    bowel = r'(?:bh|bowel|bowels|stool|stools|defecation|defaecation)'
    habit = r'(?:habit|habits|freq|frequency|freq\w*)'
    bowelhabit = r'b\.habit'
    change = r'(?:change|chang\w*|variable|irregular|irreg\w*|inc|increase|increas\w*|alt|altered|alter\w*|erratic|errat\w*)'

    # Change - bowel - (habit)
    pat_a = change + '( )' + bowel + '(?:( )' + habit + ')?'
    pat_b = change + '( )' + bowelhabit
    pat0 = '(?:' + pat_a + '|' + pat_b + ')'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bowelhabit', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # Change - habit - bowel
    pat0 = change + '( )' + habit + '( )' + bowel
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bowelhabit', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # Bowel - (habit) - change/disturbance/urgency
    du = r'(?:disturbance|disturbed|disturb\w*|urgency|urgenc\w*)'
    pat0 = bowel + '( )' + '(?:' + habit + ')?' + '(?:' + change + '|' + du + ')'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bowelhabit', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # Bowel - more often
    pat0 = bowel + '( )' + r'(?:more often|less often)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bowelhabit', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # Softer/harder stool
    pat0 = r'(?:softer|harder)( )(?:stool|stools|fecal|faecal|faeces|feces)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bowelhabit', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # Stool softer/harder
    pat0 = r'(?:stool|stools|fecal|faecal|faeces|feces)( )(?:softer|harder)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bowelhabit', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # Urgency
    pat0 = r'(?:urgency|urgenc\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bowelhabit', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Inflammation ====
    pat0 = r'(?:inflammation|inflam\w*|infl|ibd|ibs|uc|crohn|crohns|crohn\w*|colitis)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['inflam', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Elevated platelets (thrombocythaemia) ====
    pat0 = r'(?:thrombocythaemia|thrombocytosis|thrombocyt\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['thrombo', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    pat0 = r'(high|raise|raised|rise|rising|elevated|increased|increas\w*|inc\w*)( )(?:platelets|platelet|plts|plt|plat\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['thrombo', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    pat0 = r'(?:platelets|platelet|plts|plt|plat\w*)( )(high|raise|raised|rise|rising|elevated|increased|increas\w*|inc\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['thrombo', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Fatigue ====
    pat0 = r'(?:fatigue|fatig\w*|tiredness|tired\w*|tatt|lethargy|lethargia|lethar\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['fatigue', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Rectal pain ====

    # rectal pain
    pat0 = r'(?:anal|anus|rectum|rectal|perineum|perineal)( )(?:ache|aches|aching|cramp|cramps|cramping|crampy|discomfort|pain|pains|sore|soreness)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['rectalpain', pat0, pat, gap_ex]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # pain rectum
    pat0 = r'(?:ache|aches|aching|cramp|cramps|cramping|crampy|discomfort|pain|pains|sore|soreness)( )(?:anal|anus|rectum|rectal|perineum|perineal)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['rectalpain', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # tenesmus/proctalgia
    pat0 = r'(?:tenesmus|proctalgia)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['rectalpain', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Weight loss ====
    pat0 = r'(?:weight|wt)( )(?:loss|lost|losing|losed)|(?:loss|lost|losing|losed)( )(?:weight|wt)|(?:wl|wloss|wtloss)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['wl', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Bloating ====
    pat0 = r'(?:bloated|bloating|bloat\w*)'
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['bloat', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Family history of CRC ====
    # NB - currently phrases like 'no symptoms fhx crc' are marked for exclusion

    # Blocks
    fam = r'(?:sister|mother|brother|father|maternal|paternal|family|siste\w*|mothe\w*|brothe\w*|fathe\w*|matern\w*|patern\w*|famil\w*)'
    hist = r'(?:hx|history|histor\w*)'
    site = r'(?:colon|colon\w*|colorectal|colorect\w*|caecal|cecal|colic|sigmoid|sigmoid\w*|rectosigmoid|rectosigm\w*|anal|anus|rectal|rectum|bowel|bowel\w*|abdominal|abdom\w*)'
    tum = r'(?:tumour|tumor|tumo\w*|carcinoma|carcinom\w*|neoplasm|neoplas\w*|cancer|cancer\w*|ca)'

    # Combined patterns for colorectal cancer and family history
    crc = '(?:(?:' + tum + '( )' + site + ')|(?:' + site + '( )' + tum + ')|(?:crc|bowelsca))'
    fhx = '(?:(?:' + fam + '( )' + hist + ')|(?:fhx|fh))'

    # family history ~ colorectal cancer
    pat0 = fhx + '( )' + crc
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['fh', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # history ~ colorectal cancer ~ family
    pat0 = hist + '( )' + crc + '( )' + fam
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['fh', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # family ~ colorectal cancer, colorectal cancer ~ family
    pat0 = fam + '( )' + crc
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['fh', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # colorectal cancer ~ family
    pat0 = crc + '( )' + fam
    pat = update_pat(pat0, spell, gap, trim_spell=trim_spell, reorder_groups=reorder_groups)
    tmp = pd.DataFrame([['fh', pat0, pat, gap_ex_three]], columns=['category', 'pat0', 'pat', 'ex_gap'])
    dfpat = pd.concat(objs=[dfpat, tmp], axis=0)

    # ==== Add patterns for excluding negated targets ====

    # Negation patterns to exclude on left and right
    #   |(?:post)
    ex_left = r'(?:unclear|uncertain)|(?:clear|free) (?:of|from)|(?:nil|no|not|without)|no (?:evidence|features|indication|sign)|not (?:contain|indicat|represent|show|suggest)|cannot see|absence of|(?:is|are) negative for'
    ex_right = r'\: (?:no|none|negative)|(?:absent|excluded|free|negative|resolved|ruled out)|not (demonstrated|identified|indicated|known|present|seen|significant|suggested)|cannot be seen'

    # Reorder groups
    ex_left = update_pat(ex_left, spell=None, gap=None, reorder_groups=True)
    ex_right = update_pat(ex_right, spell=None, gap=None, reorder_groups=True)

    # Constrain as word
    ex_left = r'\b(?:' + ex_left + r')\b'
    ex_right = r'\b(?:' + ex_right + r')\b'

    # Add additional rules that break negation, similarly to ConText algorithm
    stop_left = r'\b(?:although|apart|aside|but|except|if|however|nevertheless|for|from|still|though|yet|which|with|that|now|always)\b'
    stop_right = r'\b(?:although|apart|aside|but|except|if|however|nevertheless|for|from|still|though|yet|which|with|that|now|always)\b'
    tdist = ''
    pre = '(?<!' + stop_right + '.{,' + str(tdist) + '}' + ')'
    post = '(?!' + '.{,' + str(tdist) + '}' + stop_left + ')'
    ex_left = ex_left + post
    ex_right = pre + ex_right

    # Constrain distance for negation patterns - must occur up to 20 char to the left or 5 to the right
    ex_left = ut.constrain_distance(ex_left, side='left', char=r'[ \w\t\:]', distance=20)
    ex_right = ut.constrain_distance(ex_right, side='right', char=r'[ \w\t\:]', distance=5)

    # Store
    dfpat['ex_left'] = ex_left
    dfpat['ex_right'] = ex_right

    dfpat = dfpat.reset_index(drop=True)

    # Simple test
    # print(re.sub(ex_left, '<detected>', 'there is no'))
    # print(re.sub(ex_left, '<detected>', 'there is no condition X but there is'))
    
    return dfpat


def extract_symptoms(df: pd.DataFrame, dfpat: pd.DataFrame, col: str = 'clind_all',
                     categories: list = None, remove_marked: bool = True):
    """Extract symptoms
        df: DataFrame that contains free text from which symptoms are extracted
        dfpat: DataFrame that contains patterns for extracting symptoms
        col: column name that contains free text
        categories: which symptom categories to extract; if None - all
        remove_marked: remove matches marked for exclusion (e.g. negated matches)
    """
    tic = time.time()
    matches = pd.DataFrame()
    flags = re.IGNORECASE | re.DOTALL
    
    if categories is None:
        categories = dfpat.category.unique()
    
    for i, row in dfpat.loc[dfpat.category.isin(categories)].iterrows():
        print('\nExtracting matches for category: {}'.format(row.category))
        m = ut.extract(df, col, row.pat, flags=flags)
        print('Number of matches: {}'.format(m.shape[0]))
        m['exclusion_indicator'] = 0
        m['exclusion_reason'] = ''

        # If gap contains unwanted keywords, mark matches for exclusion 
        groups = m.columns[m.columns.str.startswith('target_group')]
        if not groups.empty:
            for g in groups:
                print('Marking patterns for exlusion based on gap: {}'.format(g))
                msub = ut.process(m, g, [row.ex_gap], action='keep', flags=flags)
                m.loc[m.index.isin(msub.index), 'exclusion_indicator'] = 1
                m.loc[m.index.isin(msub.index), 'exclusion_reason'] += 'exclusion keywords in gap between words;'

        # If left and right side contain unwanted keywords, mark matches for exclusion
        if row.ex_left:
            msub = ut.process(m, 'left', [row.ex_left], action='keep', flags=flags)
            m.loc[m.index.isin(msub.index), 'exclusion_indicator'] = 1
            m.loc[m.index.isin(msub.index), 'exclusion_reason'] += 'negation keywords on the left;'
        if row.ex_right:
            msub = ut.process(m, 'right', [row.ex_right], action='keep', flags=flags)
            m.loc[m.index.isin(msub.index), 'exclusion_indicator'] = 1
            m.loc[m.index.isin(msub.index), 'exclusion_reason'] += 'negation keywords on the right;'
        m['category'] = row.category
        print('Number of matches marked for exclusion: {}'.format(m.loc[m.exclusion_indicator == 1].shape[0]))
        matches = pd.concat(objs=[matches, m], axis=0)  

    # Get length
    matches['len_target'] = matches.target.str.len()
    toc = time.time()
    
    # Process
    check_rm = matches.loc[matches.exclusion_indicator == 1]
    if remove_marked:
        matches = matches.loc[matches.exclusion_indicator == 0]

    # Simplify output
    group_cols = matches.columns[matches.columns.str.startswith('target_group')]
    matches = matches.drop(labels=group_cols, axis=1).reset_index(drop=True)
    check_rm = check_rm.drop(labels=group_cols, axis=1).reset_index(drop=True)
    print('Time elapsed: {:.2f} minutes'.format((toc-tic)/60))

    return matches, check_rm


def add_symptom_indicators(df, matches):
    """Add symptom indicators to original dataframe where symptoms were extracted from"""

    df['row'] = np.arange(df.shape[0])
    for c in matches.category.unique():
        submatches = matches.loc[matches.category == c]
        indcol = 'symptom_' + c
        df[indcol] = 0
        df.loc[df.row.isin(submatches.row), indcol] = 1

    return df
