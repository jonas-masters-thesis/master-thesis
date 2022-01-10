import logging

from sklearn.base import TransformerMixin

from shared.Argument import Argument
from shared.myutils import cosine_similarity


class CohesionScorer(TransformerMixin):
    """
    Provides functionality to compute the cohesion of sentences of an argument.
    """

    def __init__(self):
        # Future: Provide other measures
        self.dissimilarity = lambda u, v: 1 - cosine_similarity(u, v)
        self.log = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for argument in X:
            self.cohesion(argument)
            if len(argument.sentences) != len(argument.cohesion):
                self.log.warning(f'Number of separation scores does not match the with the number of sentences '
                                 f'({len(argument.sentences), argument.cohesion})')

        return X

    def cohesion(self, argument: Argument):
        """
        Computes the cohesion of the given argument based on the dissimilarity measure defined for this instance.

        Sums up all dissimilarities of a sentence to all sentences of the same argument, then, normalizes it by the
        number of sentence minus one.
        :param argument: argument for which cohesion should be computed
        """
        for focal_sentence in argument.sentence_embeddings:
            sum_of_dissimilarities = 0
            for other_sentence in argument.sentence_embeddings:
                sum_of_dissimilarities += self.dissimilarity(focal_sentence, other_sentence)
            argument.cohesion.append(sum_of_dissimilarities / (len(argument.sentences) - 1))
