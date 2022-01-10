import numpy as np
from sklearn.base import TransformerMixin


class SilhouetteScorer(TransformerMixin):
    """
    Computes silhouette scores according to [1] for each sentence of an argument.

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    """

    def __init__(self, limit: int = 2):
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for argument in X:
            for a_i, b_i in zip(argument.cohesion, argument.separation):
                s_i = (b_i - a_i) / max(a_i, b_i)
                argument.silhouette.append(s_i)
        return X

    def predict(self, X):
        self.transform(X)  # TODO: best-rank extract estimator
        for argument in X:
            # take the sentence indices ordered by score, but only as much as defined by limit
            idx = np.argsort(argument.silhouette)[-self.limit:]
            # excerpts should preserve sentence ordering form original
            idx = sorted(idx)

            excerpt = list()
            for i in idx:
                excerpt.append(argument.sentences[i])

            argument.excerpt = excerpt
        return X
