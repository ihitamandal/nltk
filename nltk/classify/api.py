# Natural Language Toolkit: Classifier Interface
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Interfaces for labeling tokens with category labels (or "class labels").

``ClassifierI`` is a standard interface for "single-category
classification", in which the set of categories is known, the number
of categories is finite, and each text belongs to exactly one
category.

``MultiClassifierI`` is a standard interface for "multi-category
classification", which is like single-category classification except
that each text belongs to zero or more categories.
"""
from functools import lru_cache
from typing import List, Type

from nltk.internals import overridden

##//////////////////////////////////////////////////////
# { Classification Interfaces
##//////////////////////////////////////////////////////


class ClassifierI:
    """
    A processing interface for labeling tokens with a single category
    label (or "class").  Labels are typically strs or
    ints, but can be any immutable type.  The set of labels
    that the classifier chooses from must be fixed and finite.

    Subclasses must define:
      - ``labels()``
      - either ``classify()`` or ``classify_many()`` (or both)

    Subclasses may define:
      - either ``prob_classify()`` or ``prob_classify_many()`` (or both)
    """

    def labels(self):
        """
        :return: the list of category labels used by this classifier.
        :rtype: list of (immutable)
        """
        raise NotImplementedError()

    def classify(self, featureset):
        """
        :return: the most appropriate label for the given featureset.
        :rtype: label
        """
        if overridden(self.classify_many):
            return self.classify_many([featureset])[0]
        else:
            raise NotImplementedError()

    def prob_classify(self, featureset):
        """
        :return: a probability distribution over labels for the given
            featureset.
        :rtype: ProbDistI
        """
        if overridden(self.prob_classify_many):
            return self.prob_classify_many([featureset])[0]
        else:
            raise NotImplementedError()

    def classify_many(self, featuresets):
        """
        Apply ``self.classify()`` to each element of ``featuresets``.  I.e.:

            return [self.classify(fs) for fs in featuresets]

        :rtype: list(label)
        """
        return [self.classify(fs) for fs in featuresets]

    def prob_classify_many(self, featuresets):
        """
        Apply ``self.prob_classify()`` to each element of ``featuresets``.  I.e.:

            return [self.prob_classify(fs) for fs in featuresets]

        :rtype: list(ProbDistI)
        """
        return [self.prob_classify(fs) for fs in featuresets]


class MultiClassifierI:
    """
    A processing interface for labeling tokens with zero or more
    category labels (or "labels").  Labels are typically strs
    or ints, but can be any immutable type.  The set of labels
    that the multi-classifier chooses from must be fixed and finite.

    Subclasses must define:
      - ``labels()``
      - either ``classify()`` or ``classify_many()`` (or both)

    Subclasses may define:
      - either ``prob_classify()`` or ``prob_classify_many()`` (or both)
    """

    def labels(self):
        """
        :return: the list of category labels used by this classifier.
        :rtype: list of (immutable)
        """
        raise NotImplementedError()

    def classify(self, featureset: object) -> set:
        """Return the most appropriate set of labels for the given featureset.

        Parameters
        ----------
        featureset : object
            The feature set for classification.

        Returns
        -------
        set
            The set of labels for the featureset.

        Raises
        ------
        NotImplementedError
            If the classify_many method is not overridden in the subclass.
        """
        if overridden(self.classify_many):
            return self.classify_many([featureset])[0]
        else:
            raise NotImplementedError()

    def prob_classify(self, featureset):
        """
        :return: a probability distribution over sets of labels for the
            given featureset.
        :rtype: ProbDistI
        """
        if overridden(self.prob_classify_many):
            return self.prob_classify_many([featureset])[0]
        else:
            raise NotImplementedError()

    def classify_many(self, featuresets: List[object]) -> List[set]:
        """Apply classify method to each element of featuresets.

        Parameters
        ----------
        featuresets : List[object]
            The list of feature sets for classification.

        Returns
        -------
        List[set]
            The list of labeled sets for each featureset.
        """
        return [self.classify(fs) for fs in featuresets]

    def prob_classify_many(self, featuresets):
        """
        Apply ``self.prob_classify()`` to each element of ``featuresets``.  I.e.:

            return [self.prob_classify(fs) for fs in featuresets]

        :rtype: list(ProbDistI)
        """
        return [self.prob_classify(fs) for fs in featuresets]


@lru_cache(maxsize=128)
def _mro(cls: Type) -> List[Type]:
    """Compute and cache the method resolution order of the class.

    Parameters
    ----------
    cls : Type
        The class to compute the MRO for.

    Returns
    -------
    List[Type]
        A list of classes in the method resolution order.
    """
    return list(cls.__mro__)


# # [XX] IN PROGRESS:
# class SequenceClassifierI:
#     """
#     A processing interface for labeling sequences of tokens with a
#     single category label (or "class").  Labels are typically
#     strs or ints, but can be any immutable type.  The set
#     of labels that the classifier chooses from must be fixed and
#     finite.
#     """
#     def labels(self):
#         """
#         :return: the list of category labels used by this classifier.
#         :rtype: list of (immutable)
#         """
#         raise NotImplementedError()

#     def prob_classify(self, featureset):
#         """
#         Return a probability distribution over labels for the given
#         featureset.

#         If ``featureset`` is a list of featuresets, then return a
#         corresponding list containing the probability distribution
#         over labels for each of the given featuresets, where the
#         *i*\ th element of this list is the most appropriate label for
#         the *i*\ th element of ``featuresets``.
#         """
#         raise NotImplementedError()

#     def classify(self, featureset):
#         """
#         Return the most appropriate label for the given featureset.

#         If ``featureset`` is a list of featuresets, then return a
#         corresponding list containing the most appropriate label for
#         each of the given featuresets, where the *i*\ th element of
#         this list is the most appropriate label for the *i*\ th element
#         of ``featuresets``.
#         """
#         raise NotImplementedError()
