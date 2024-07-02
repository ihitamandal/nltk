# Natural Language Toolkit: CISTEM Stemmer for German
# Copyright (C) 2001-2023 NLTK Project
# Author: Leonie Weissweiler <l.weissweiler@outlook.de>
#         Tom Aarsen <> (modifications)
# Algorithm: Leonie Weissweiler <l.weissweiler@outlook.de>
#            Alexander Fraser <fraser@cis.lmu.de>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import re
from typing import Tuple

from nltk.stem.api import StemmerI


class Cistem(StemmerI):
    """
    CISTEM Stemmer for German

    This is the official Python implementation of the CISTEM stemmer.
    It is based on the paper
    Leonie Weissweiler, Alexander Fraser (2017). Developing a Stemmer for German
    Based on a Comparative Analysis of Publicly Available Stemmers.
    In Proceedings of the German Society for Computational Linguistics and Language
    Technology (GSCL)
    which can be read here:
    https://www.cis.lmu.de/~weissweiler/cistem/

    In the paper, we conducted an analysis of publicly available stemmers,
    developed two gold standards for German stemming and evaluated the stemmers
    based on the two gold standards. We then proposed the stemmer implemented here
    and show that it achieves slightly better f-measure than the other stemmers and
    is thrice as fast as the Snowball stemmer for German while being about as fast
    as most other stemmers.

    case_insensitive is a a boolean specifying if case-insensitive stemming
    should be used. Case insensitivity improves performance only if words in the
    text may be incorrectly upper case. For all-lowercase and correctly cased
    text, best performance is achieved by setting case_insensitive for false.

    :param case_insensitive: if True, the stemming is case insensitive. False by default.
    :type case_insensitive: bool
    """

    strip_ge = re.compile(r"^ge(.{4,})")
    repl_xx = re.compile(r"(.)\1")
    strip_emr = re.compile(r"e[mr]$")
    strip_nd = re.compile(r"nd$")
    strip_t = re.compile(r"t$")
    strip_esn = re.compile(r"[esn]$")
    repl_xx_back = re.compile(r"(.)\*")

    def __init__(self, case_insensitive: bool = False):
        self._case_insensitive = case_insensitive

    @staticmethod
    def replace_to(word: str) -> str:
        word = word.replace("sch", "$")
        word = word.replace("ei", "%")
        word = word.replace("ie", "&")
        word = Cistem.repl_xx.sub(r"\1*", word)

        return word

    @staticmethod
    def replace_back(word: str) -> str:
        word = Cistem.repl_xx_back.sub(r"\1\1", word)
        word = word.replace("%", "ei")
        word = word.replace("&", "ie")
        word = word.replace("$", "sch")

        return word

    def stem(self, word: str) -> str:
        """Stems the input word.

        :param word: The word that is to be stemmed.
        :type word: str
        :return: The stemmed word.
        :rtype: str

        >>> from nltk.stem.cistem import Cistem
        >>> stemmer = Cistem()
        >>> s1 = "Speicherbehältern"
        >>> stemmer.stem(s1)
        'speicherbehalt'
        >>> s2 = "Grenzpostens"
        >>> stemmer.stem(s2)
        'grenzpost'
        >>> s3 = "Ausgefeiltere"
        >>> stemmer.stem(s3)
        'ausgefeilt'
        >>> stemmer = Cistem(True)
        >>> stemmer.stem(s1)
        'speicherbehal'
        >>> stemmer.stem(s2)
        'grenzpo'
        >>> stemmer.stem(s3)
        'ausgefeil'
        """
        if len(word) == 0:
            return word

        upper = word[0].isupper()
        word = word.lower()

        word = word.replace("ü", "u")
        word = word.replace("ö", "o")
        word = word.replace("ä", "a")
        word = word.replace("ß", "ss")

        word = Cistem.strip_ge.sub(r"\1", word)

        return self._segment_inner(word, upper)[0]

    def segment(self, word: str) -> Tuple[str, str]:
        """
        This method works very similarly to stem (:func:'cistem.stem'). The difference is that in
        addition to returning the stem, it also returns the rest that was removed at
        the end. To be able to return the stem unchanged so the stem and the rest
        can be concatenated to form the original word, all subsitutions that altered
        the stem in any other way than by removing letters at the end were left out.

        :param word: The word that is to be stemmed.
        :type word: str
        :return: A tuple of the stemmed word and the removed suffix.
        :rtype: Tuple[str, str]

        >>> from nltk.stem.cistem import Cistem
        >>> stemmer = Cistem()
        >>> s1 = "Speicherbehältern"
        >>> stemmer.segment(s1)
        ('speicherbehält', 'ern')
        >>> s2 = "Grenzpostens"
        >>> stemmer.segment(s2)
        ('grenzpost', 'ens')
        >>> s3 = "Ausgefeiltere"
        >>> stemmer.segment(s3)
        ('ausgefeilt', 'ere')
        >>> stemmer = Cistem(True)
        >>> stemmer.segment(s1)
        ('speicherbehäl', 'tern')
        >>> stemmer.segment(s2)
        ('grenzpo', 'stens')
        >>> stemmer.segment(s3)
        ('ausgefeil', 'tere')
        """
        if len(word) == 0:
            return ("", "")

        upper = word[0].isupper()
        word = word.lower()

        return self._segment_inner(word, upper)

    def _segment_inner(self, word: str, upper: bool):
        """Inner method for iteratively applying the code stemming regexes.
        This method receives a pre-processed variant of the word to be stemmed,
        or the word to be segmented, and returns a tuple of the word and the
        removed suffix.

        :param word: A pre-processed variant of the word that is to be stemmed.
        :type word: str
        :param upper: Whether the original word started with a capital letter.
        :type upper: bool
        :return: A tuple of the stemmed word and the removed suffix.
        :rtype: Tuple[str, str]
        """
        rest_length = 0
        word_copy = word[:]

        word = Cistem.replace_to(word)
        rest = ""

        def apply_regex(pattern, word, rest_increment):
            new_word, n = pattern.subn("", word)
            return (new_word, rest_increment * n, n)

        while len(word) > 3:
            if len(word) > 5:
                word, increment, n = apply_regex(Cistem.strip_emr, word, 2)
                if n:
                    rest_length += increment
                    continue

                word, increment, n = apply_regex(Cistem.strip_nd, word, 2)
                if n:
                    rest_length += increment
                    continue

            if not upper or self._case_insensitive:
                word, increment, n = apply_regex(Cistem.strip_t, word, 1)
                if n:
                    rest_length += increment
                    continue

            word, increment, n = apply_regex(Cistem.strip_esn, word, 1)
            if n:
                rest_length += increment
                continue
            else:
                break

        word = Cistem.replace_back(word)

        if rest_length:
            rest = word_copy[-rest_length:]

        return (word, rest)

    @staticmethod
    def replace_to(word: str) -> str:
        replacements = {"sch": "$", "ei": "%", "ie": "&"}
        for k, v in replacements.items():
            word = word.replace(k, v)
        return Cistem.repl_xx.sub(r"\1*", word)

    @staticmethod
    def replace_back(word: str) -> str:
        word = Cistem.repl_xx_back.sub(r"\1\1", word)
        replacements = {"%": "ei", "&": "ie", "$": "sch"}
        for k, v in replacements.items():
            word = word.replace(k, v)
        return word

    def stem(self, word: str) -> str:
        """Stems the input word.

        :param word: The word that is to be stemmed.
        :type word: str
        :return: The stemmed word.
        :rtype: str

        >>> from nltk.stem.cistem import Cistem
        >>> stemmer = Cistem()
        >>> s1 = "Speicherbehältern"
        >>> stemmer.stem(s1)
        'speicherbehalt'
        >>> s2 = "Grenzpostens"
        >>> stemmer.stem(s2)
        'grenzpost'
        >>> s3 = "Ausgefeiltere"
        >>> stemmer.stem(s3)
        'ausgefeilt'
        >>> stemmer = Cistem(True)
        >>> stemmer.stem(s1)
        'speicherbehal'
        >>> stemmer.stem(s2)
        'grenzpo'
        >>> stemmer.stem(s3)
        'ausgefeil'
        """
        if not word:
            return word

        upper = word[0].isupper()
        word = word.lower()

        umlaut_replacements = {"ü": "u", "ö": "o", "ä": "a", "ß": "ss"}
        for k, v in umlaut_replacements.items():
            word = word.replace(k, v)

        word = Cistem.strip_ge.sub(r"\1", word)

        return self._segment_inner(word, upper)[0]

    def segment(self, word: str) -> Tuple[str, str]:
        """
        This method works very similarly to stem (:func:'cistem.stem'). The difference is that in
        addition to returning the stem, it also returns the rest that was removed at
        the end. To be able to return the stem unchanged so the stem and the rest
        can be concatenated to form the original word, all subsitutions that altered
        the stem in any other way than by removing letters at the end were left out.

        :param word: The word that is to be stemmed.
        :type word: str
        :return: A tuple of the stemmed word and the removed suffix.
        :rtype: Tuple[str, str]

        >>> from nltk.stem.cistem import Cistem
        >>> stemmer = Cistem()
        >>> s1 = "Speicherbehältern"
        >>> stemmer.segment(s1)
        ('speicherbehält', 'ern')
        >>> s2 = "Grenzpostens"
        >>> stemmer.segment(s2)
        ('grenzpost', 'ens')
        >>> s3 = "Ausgefeiltere"
        >>> stemmer.segment(s3)
        ('ausgefeilt', 'ere')
        >>> stemmer = Cistem(True)
        >>> stemmer.segment(s1)
        ('speicherbehäl', 'tern')
        >>> stemmer.segment(s2)
        ('grenzpo', 'stens')
        >>> stemmer.segment(s3)
        ('ausgefeil', 'tere')
        """
        if not word:
            return ("", "")

        upper = word[0].isupper()
        word = word.lower()

        return self._segment_inner(word, upper)
