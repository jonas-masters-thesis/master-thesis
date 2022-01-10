import logging

import matplotlib.pyplot as plt

from DataHandler import DataHandler
from evaluation.EdgeCorrelation import EdgeCorrelation
from evaluation.SentenceArgReAllocator import SentenceArgReAllocator
from evaluation.SilhouetteCoefficient import SilhouetteCoefficient
from myutils import make_logging


def main():
    make_logging('evaluation', level=logging.INFO)
    log = logging.getLogger(__name__)
    data = DataHandler()
    data.load_bin('../contra-lexrank/results/results.pickle')
    log.info(f'Load {len(data.get_filtered_arguments(DataHandler.get_args_filter_length()))} arguments')

    # Compute edge correlation
    ec = EdgeCorrelation('results/initial-arguments-edge-correlation.json')
    corr_original = ec.edge_correlation(data.get_arguments())
    log.info(f'Edge correlation of arguments as given (original): {corr_original}.')
    sc = SilhouetteCoefficient('results/initial-arguments-silhouette-coefficient.json')
    silh_coef = sc.silhouette_coefficient(data.get_arguments())
    log.info(f'Silhouette coefficient of arguments as given (original): {silh_coef}.')

    # Move sentences to closest centroid and re-compute edge correlation
    reallocator = SentenceArgReAllocator()
    reallocator.prepare_snippet_embeddings(data.get_filtered_arguments(DataHandler.get_args_filter_length()))
    reallocator.re_allocate(data.get_arguments())
    new_arguments = reallocator.convert_to_argument()
    for a in new_arguments:
        log.debug(f'{a.arg_id} has {len(a.sentence_embeddings)} sentences.')

    # Compute edge correlation of new arguments
    ec = EdgeCorrelation('results/reallocated-arguments-edge-correlation.json')
    corr_realloc = ec.edge_correlation(new_arguments)
    log.info(f'Edge correlation of arguments after re-allocation: {corr_realloc}.')
    sc = SilhouetteCoefficient('results/reallocated-arguments-silhouette-coefficient.json')
    silh_coef_realloc = sc.silhouette_coefficient(new_arguments)
    log.info(f'Silhouette coefficient of arguments as given (original): {silh_coef_realloc}.')

    log.debug('Finished.')


def plot_sim_heat(ec: EdgeCorrelation):
    plt.imshow(ec.similarity_matrix, cmap='hot')
    plt.show()


if __name__ == '__main__':
    main()
