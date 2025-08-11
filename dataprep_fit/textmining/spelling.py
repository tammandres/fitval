"""Helper functions for simple spell checks"""
import regex as re


def edits1(word):
    """All edits that are one edit away from `word`.
    Taken from Peter Norvig, https://norvig.com/spell-correct.html
    """
    letters = 'abcdefghijklmnopqrstuvwxyz;,.\/-: 0123456789'  # Include digits and some other chars
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """All edits that are two edits away from `word`.
    Taken from Peter Norvig, modified, https://norvig.com/spell-correct.html
    """
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


def existing_variations(word, text, combine_edits=True):
    """Get variations of 'word' that exist in 'text' excluding the original word
    Args: word [str], text [str], two_edits [bool]
    Based on Peter Norvig, https://norvig.com/spell-correct.html
    """
    words = set(re.findall(r'\w+[;,\.\\\/:\w]*\w+', text.lower()))
    var1 = sorted([e for e in edits1(word) if e in words])
    var2 = sorted([e for e in edits2(word) if e in words])

    var1 = [v for v in var1 if v != word]
    var2 = [v for v in var2 if v not in var1 and v != word]
    if combine_edits:
        var = sorted(set(var1 + var2))
        return var
    else:
        return var1, var2


def existing_variations_re(pat, text):
    """Get variations of 'word' that exist in 'text'
    Excluding the original word.
    Args: word [str], text [str], two_edits [bool]
    Based on Peter Norvig, https://norvig.com/spell-correct.html
    """
    words = set(re.findall(r'\w+[;,\.\\\/:\w]*\w+', text.lower()))
    wpat = set(re.findall(pat, text.lower(), flags=re.IGNORECASE))
    if wpat:
        print('\nWords that match pattern: {}'.format(wpat))
    else:
        print('\nNo words match that pattern')
    res = {}
    var1, var2 = [], []
    for word in wpat:
        var1.extend([e for e in edits1(word) if e in words])
        var2.extend([e for e in edits2(word) if e in words])
        # var1  = sorted([e for e in edits1(word) if e in words])
        # var2  = sorted([e for e in edits2(word) if e in words])
        # var1  = [v for v in var1 if v != word]
        # var2  = [v for v in var2 if v not in var1 and v != word]
        # res[word] = {'edits1':var1,'edits2':var2}
    var1 = sorted(set([v for v in var1 if v not in wpat]), key=len)
    var2 = sorted(set([v for v in var2 if v not in var1 and v not in wpat]), key=len)
    res['edits1'] = var1
    res['edits2'] = var2
    return res


def add_var(words, spell):
    wspell = spell.repl.to_list()
    for w in list(words):
        if w in wspell:
            padd = spell.loc[spell['repl'] == w, 'pat'].iloc[0]
            wadd = padd.split('|')
            words += wadd
    return words


def update_pat(pat, spell=None, gap=None, reorder_groups=False, trim_spell=True, constrain_as_word=True,
               trim_len=7):
    """
    pat is a regex that contains groups of words separated by '|'
    spell is the spellcheck table with columns 'pat' and 'repl'

    Caveats:
        * pat must contain whole words like 'black|cat|magic' rather than regexes such as 'bla?ck|cat\w*|magic\w*'
        * reorder_groups works if there are no nested groups, and if groups are non-capture, e.g. '(?:cat|black)'
    """

    # Add spelling variations for words in pattern
    #  e.g. 'abdo' -> 'abdo|abdominal|abd|abdpo|abdso|abo|adbo' ...
    if spell is not None:
        words = re.findall(r'\w+', pat)  # Find all words present in pat
        wspell = spell.repl.to_list()  # Words for which spelling variations are available
        for w in words:
            if w in wspell:
                padd = spell.loc[spell['repl'] == w, 'pat'].iloc[0]
                if trim_spell:
                    wadd = padd.split('|')
                    wadd = list(set([w[0:trim_len] + r'\w*' if len(w) > trim_len else w for w in wadd]))
                    padd = '|'.join(wadd)
                padd = w + '|' + padd  # expand word with additional words in spellcheck table
                pat = re.sub(r'\b' + w + r'\b', padd, pat, flags=re.IGNORECASE | re.DOTALL)

    # Reorder non-capture groups by length
    if reorder_groups:
        # pat = re.sub('((?<!\()(?<!\(\?\:)\\b(?:\w+\|)*\w+\\b(?!\)))', '(?:\g<1>)', pat)

        # Detect non-capture groups w at least one wordchar to avoid empty cap group ( )
        groups = re.findall(r'\(\?\:[^\(]*?\w.*?\)', pat)
        for g in groups:
            g2 = str(g)
            g2 = re.sub(r'\(\??\:?', '', g2)
            g2 = re.sub(r'\)', '', g2)
            w2 = g2.split('|')
            w2 = sorted(w2, key=len, reverse=True)
            g2 = '(?:' + '|'.join(w2) + ')'
            g = re.escape(g)
            pat = re.sub(g, g2, pat)

    # Update gap
    if gap is not None:
        pat = re.sub(' ', gap, pat)

    if constrain_as_word:
        pat = r'\b(?:' + pat + r')\b'

    return pat
