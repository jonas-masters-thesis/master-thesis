import json
import logging
from typing import List

import numpy as np
from sklearn.metrics import silhouette_score

from Argument import Argument


class SilhouetteCoefficient:

    def __init__(self, save_path=None):
        self.log = logging.getLogger(SilhouetteCoefficient.__name__)
        self.save_path = save_path

    def silhouette_coefficient(self, arguments: List[Argument]):
        contexts = set([a.query for a in arguments])
        result = dict()
        for context in contexts:
            contextual_arguments = list(filter(lambda a: a.query == context, arguments))
            self.log.debug(f'{len(contextual_arguments)} arguments in context {context}.')

            X = list()
            labels = list()
            for i, argument in enumerate(contextual_arguments):
                for sentence in argument.sentence_embeddings:
                    X.append(sentence)
                    labels.append(i)

            assert len(X) == len(labels)
            if len(set(labels)) > 1:
                result[context] = silhouette_score(X, labels, metric='cosine')
            else:
                self.log.warning(f'Context {context} has only one argument.')
                result[context] = np.NAN

        if self.save_path is not None:
            self.save_results(result)

        return result

    def save_results(self, result):
        result = {k: float(v) for k, v in result.items()}  # Convert from numpy's float to python float
        with open(self.save_path, 'w', encoding='utf-8') as file:
            json.dump(result, file)
