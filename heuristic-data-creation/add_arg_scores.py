import logging
import pickle

import torch

from ArgumentativenessScorer import ArgumentativenessScorer
from FeaturedArgument import FeaturedArgument
from myutils import make_logging

log = logging.getLogger(__name__)


def main():
    make_logging('add_arg_scores')

    with open('../../not-gitted/filtered-w-reference-snippets-a-features-r_0.1-args-me[0-100].json.pickle',
              'rb') as f:
        d = pickle.load(f)

    X = list()
    for e in d:
        X.append(FeaturedArgument.from_argsme_json(e))

    scorer = ArgumentativenessScorer(calculation=None,
                                     discourse_markers='C:/Users/Jonas/git/thesis/code/contra-lexrank/discourse'
                                                       '-markers.txt',
                                     claim_lexicon='C:/Users/Jonas/git/thesis/code/contra-lexrank/ClaimLexicon.txt')
    scorer.transform(X)

    for arg in X:
        scores = arg.argumentativeness_scores[0]
        arg.surface_features = torch.cat((arg.surface_features, torch.tensor(scores).unsqueeze(1)), 1)
        arg.number_of_surface_features = 6

    with open(
            '../../not-gitted/filtered-w-reference-snippets-a-features-incl-argscores-r_0.1-args-me[0-100].json.pickle',
            'wb') as f:
        pickle.dump(X, f)


if __name__ == '__main__':
    main()
