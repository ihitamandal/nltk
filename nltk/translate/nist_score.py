# Natural Language Toolkit: NIST Score
#
# Copyright (C) 2001-2023 NLTK Project
# Authors:
# Contributors:
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""NIST score implementation."""

import fractions
import math
from collections import Counter

from nltk.util import ngrams


def sentence_nist(references, hypothesis, n=5):
    """
    Calculate NIST score from
    George Doddington. 2002. "Automatic evaluation of machine translation quality
    using n-gram co-occurrence statistics." Proceedings of HLT.
    Morgan Kaufmann Publishers Inc. https://dl.acm.org/citation.cfm?id=1289189.1289273

    DARPA commissioned NIST to develop an MT evaluation facility based on the BLEU
    score. The official script used by NIST to compute BLEU and NIST score is
    mteval-14.pl. The main differences are:

     - BLEU uses geometric mean of the ngram overlaps, NIST uses arithmetic mean.
     - NIST has a different brevity penalty
     - NIST score from mteval-14.pl has a self-contained tokenizer

    Note: The mteval-14.pl includes a smoothing function for BLEU score that is NOT
          used in the NIST score computation.

    >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...               'ensures', 'that', 'the', 'military', 'always',
    ...               'obeys', 'the', 'commands', 'of', 'the', 'party']

    >>> hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
    ...               'forever', 'hearing', 'the', 'activity', 'guidebook',
    ...               'that', 'party', 'direct']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...               'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...               'heed', 'Party', 'commands']

    >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...               'guarantees', 'the', 'military', 'forces', 'always',
    ...               'being', 'under', 'the', 'command', 'of', 'the',
    ...               'Party']

    >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...               'army', 'always', 'to', 'heed', 'the', 'directions',
    ...               'of', 'the', 'party']

    >>> sentence_nist([reference1, reference2, reference3], hypothesis1) # doctest: +ELLIPSIS
    3.3709...

    >>> sentence_nist([reference1, reference2, reference3], hypothesis2) # doctest: +ELLIPSIS
    1.4619...

    :param references: reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param n: highest n-gram order
    :type n: int
    """
    return corpus_nist([references], [hypothesis], n)


def corpus_nist(list_of_references, hypotheses, n=5):
    """
    Calculate a single corpus-level NIST score (aka. system-level BLEU) for all
    the hypotheses and their respective references.

    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param n: highest n-gram order
    :type n: int
    """
    assert len(list_of_references) == len(
        hypotheses
    ), "The number of hypotheses and their reference(s) should be the same"

    ngram_freq = Counter()
    total_reference_words = sum(len(ref) for refs in list_of_references for ref in refs)

    for references in list_of_references:
        for reference in references:
            for i in range(1, n + 1):
                ngram_freq.update(ngrams(reference, i))

    information_weights = {
        ngram: (
            math.log(
                float(ngram_freq[ngram[:-1]]) / ngram_freq[ngram]
                if ngram[:-1] in ngram_freq
                else total_reference_words / ngram_freq[ngram],
                2,
            )
        )
        for ngram in ngram_freq
    }

    nist_precision_numerator_per_ngram = Counter()
    nist_precision_denominator_per_ngram = Counter()
    l_ref, l_sys = 0, 0

    for i in range(1, n + 1):
        for references, hypothesis in zip(list_of_references, hypotheses):
            hyp_len = len(hypothesis)
            nist_score_per_ref = []

            hyp_ngrams_i = Counter(ngrams(hypothesis, i)) if hyp_len >= i else Counter()

            for reference in references:
                ref_len = len(reference)
                ref_ngrams_i = (
                    Counter(ngrams(reference, i)) if ref_len >= i else Counter()
                )
                ngram_overlaps = hyp_ngrams_i & ref_ngrams_i

                numerator = sum(
                    information_weights[ng] * count
                    for ng, count in ngram_overlaps.items()
                )
                denominator = sum(hyp_ngrams_i.values())
                precision = 0 if denominator == 0 else numerator / denominator
                nist_score_per_ref.append((precision, numerator, denominator, ref_len))

            precision, numerator, denominator, ref_len = max(nist_score_per_ref)
            nist_precision_numerator_per_ngram[i] += numerator
            nist_precision_denominator_per_ngram[i] += denominator
            l_ref += ref_len
            l_sys += hyp_len

    nist_precision = sum(
        nist_precision_numerator_per_ngram[i] / nist_precision_denominator_per_ngram[i]
        for i in nist_precision_numerator_per_ngram
    )

    return nist_precision * nist_length_penalty(l_ref, l_sys)


def nist_length_penalty(ref_len, hyp_len):
    """
    Calculates the NIST length penalty, from Eq. 3 in Doddington (2002)

        penalty = exp( beta * log( min( len(hyp)/len(ref) , 1.0 )))

    where,

        `beta` is chosen to make the brevity penalty factor = 0.5 when the
        no. of words in the system output (hyp) is 2/3 of the average
        no. of words in the reference translation (ref)

    The NIST penalty is different from BLEU's such that it minimize the impact
    of the score of small variations in the length of a translation.
    See Fig. 4 in  Doddington (2002)
    """
    ratio = hyp_len / ref_len
    if 0 < ratio < 1:
        ratio_x, score_x = 1.5, 0.5
        beta = math.log(score_x) / math.log(ratio_x) ** 2
        return math.exp(beta * math.log(ratio) ** 2)
    else:  # ratio <= 0 or ratio >= 1
        return max(min(ratio, 1.0), 0.0)
