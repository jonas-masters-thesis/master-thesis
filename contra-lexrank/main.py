import logging

from sklearn.pipeline import Pipeline

from ArgumentativenessScorer import ArgumentativenessScorer
from CentralityScorer import CentralityScorer
from ContraLexRank import ContraLexRank
from ContrastivenessScorer import ContrastivenessScorer
from myutils import make_logging
from shared.DataHandler import DataHandler


def main():
    make_logging('lexrank')
    log = logging.getLogger(__name__)

    data = DataHandler()
    data.load_bin('../../not-gitted/dataset_as_json_file.pickle')  # embeddings are pre-computed
    X = data.get_filtered_arguments([DataHandler.get_args_filter_length()])

    pipeline = Pipeline(steps=[
        ('argumentativeness', ArgumentativenessScorer()),
        ('contrastiveness', ContrastivenessScorer()),
        ('centrality', CentralityScorer()),
        ('clr', ContraLexRank(0.7, 0.5, 0.2)),
    ])
    pipeline.predict(X)

    # compute trade-off (ssc, soc)
    # trade_off = TradeOffScorer()
    # trade_off.transform(X)

    data.save_results('results/results.json')
    data.dump_data('results/results.pickle')

    log.info('Finished.')


if __name__ == '__main__':
    main()

# def gs(pipe, X):
#     grid = GridSearchCV(
#         estimator=pipe,
#         param_grid={
#             'clr__alpha': [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
#         },
#         scoring=['accuracy', 'f1', 'precision', 'recall'],
#         refit=False
#     )
#     search_results = grid.fit(X, None)
#     pd.DataFrame(data=search_results.cv_results_)[
#         ['rank_test_f1', 'mean_test_f1', 'std_test_f1', 'mean_test_precision', 'std_test_precision',
#         'mean_test_recall',
#          'std_test_recall', 'mean_test_accuracy', 'std_test_accuracy', 'param_penalty', 'param_C']].sort_values(
#         by='rank_test_f1')
