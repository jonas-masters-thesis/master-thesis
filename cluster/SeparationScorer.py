import logging
from typing import List

from sklearn.base import TransformerMixin

from shared.Argument import Argument
from shared.DataHandler import DataHandler
from shared.myutils import cosine_similarity


class SeparationScorer(TransformerMixin):
    """
    Provides functionality to compute the separation of sentences of an argument.
    """

    def __init__(self):
        # Future: Provide other measures
        self.dissimilarity = lambda u, v: 1 - cosine_similarity(u, v)
        self.log = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        last = ('', '')
        for argument in X:
            if argument.query == last[0]:
                context = last[1]
            else:
                context = DataHandler.get_query_context(X, argument.query)
                last = (argument.query, context)
            self.separation(argument, context)
            if len(argument.sentences) != len(argument.separation):
                self.log.warning(f'Number of separation scores does not match the with the number of sentences '
                                 f'({len(argument.sentences), argument.separation})')

        return X

    def separation(self, argument: Argument, context: List[Argument]):
        """
        Computes the separation of the given argument based on the dissimilarity measure defined for this instance.

        Sums up all dissimilarities of a sentence to all sentences of the all other arguments' sentences, then,
        normalizes it by the number of sentence.
        :param argument: argument for which separation should be computed
        :param context: arguments that form the other clusters (all other arguments form one large cluster)
        """
        for focal_sentence in argument.sentence_embeddings:
            sum_of_dissimilarities = 0
            _counter = 0
            for other_argument in context:
                for other_sentence in other_argument.sentence_embeddings:
                    sum_of_dissimilarities += self.dissimilarity(focal_sentence, other_sentence)
                    _counter += 1
            argument.separation.append(sum_of_dissimilarities / _counter)
