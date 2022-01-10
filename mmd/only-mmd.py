import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline

from Argument import Argument
from DataHandler import DataHandler
from EdgeCorrelation import EdgeCorrelation
from FeaturedArgument import FeaturedArgument
from Inference import Inference
from MMDBase import MMDBase
from SentenceArgReAllocator import SentenceArgReAllocator
from SilhouetteCoefficient import SilhouetteCoefficient
from TradeOffScorer import TradeOffScorer
from myutils import make_logging, tokenize

exp_id = 'MMD-01'
log = logging.getLogger(__name__)

DATA_PATH = '../../not-gitted/argsme-crawled/1629700068.9873986-6578-arguments.pickle'

if torch.cuda.device_count() >= 1:
    DEVICE = torch.cuda.current_device()
    log.info(torch.cuda.current_device())
    log.info(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    DEVICE = -1
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # ), cache_dir='cache')
model = BertForSequenceClassification.from_pretrained('../bert-finetuning/results/argQ-bert-base-uncased',
                                                      local_files_only=True, cache_dir='cache')
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', task='ArgQ')
sim_func = MMDBase(param_gamma=.0, param_lambda=.0).cosine_kernel_matrix


def main():
    make_logging('mmd')

    data = get_data(DATA_PATH)
    context_lvl, arg_lvl = baseline_scores(data)

    global results
    results = context_lvl
    del context_lvl
    global arg_level_results
    arg_level_results = arg_lvl
    del arg_lvl

    # Special variant where surface features do not count.
    thetaAB = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float64)

    lambdas = [.125, .25, .375, .5, .625, .75, .875]
    counter = 0
    for lambda_ in lambdas:
        log.info(f'Running param combo {counter}/{len(lambdas)}, lambda={lambda_}')
        counter += 1
        summarizer = Inference(param_gamma=0., param_lambda=lambda_, snippet_length=2, thetaA=thetaAB, thetaB=thetaAB)
        summarizer.fit(data)
        evaluation(data, lambda_)
        arg_level_eval(data, lambda_)
        results.to_csv(f'results/{exp_id}.csv', index=False)
        arg_level_results.to_csv(f'results/{exp_id}-arg-level.csv', index=False)


def evaluation(X: List[FeaturedArgument], lambda_):
    log.info('Starting context-level evaluation...')
    # Move sentences to closest centroid and re-compute edge correlation
    reallocator = SentenceArgReAllocator()
    reallocator.prepare_snippet_embeddings(X)
    reallocator.re_allocate(X)
    new_arguments = reallocator.convert_to_argument()

    # Compute edge correlation of new arguments
    ec = EdgeCorrelation()
    corr_realloc = ec.edge_correlation(new_arguments)
    log.info(f'Edge correlation of arguments after re-allocation: {corr_realloc}.')
    sc = SilhouetteCoefficient()
    silh_coef_realloc = sc.silhouette_coefficient(new_arguments)
    log.info(f'Silhouette coefficient of arguments after re-allocation: {silh_coef_realloc}.')

    current_results = {'lambda': lambda_}
    correlations = list()
    coefficients = list()
    for context in DataHandler.get_query_context_keys(X):
        current_results[f'{context}_edge_corr'] = corr_realloc[context].correlation
        correlations.append(corr_realloc[context].correlation)
        current_results[f'{context}_silhouette_coef'] = silh_coef_realloc[context]
        coefficients.append(silh_coef_realloc[context])

    current_results['avg_edge_corr'] = np.mean(np.array(correlations))
    current_results['avg_silhouette_coef'] = np.mean(np.array(coefficients))

    global results
    results = results.append(current_results, ignore_index=True)


def arg_level_eval(arguments: List[FeaturedArgument], lambda_):
    log.info('Starting argument-level evaluation...')
    scorer = TradeOffScorer()
    scorer.transform(arguments)
    records = list()
    for arg in arguments:
        sim_mat = sim_func(torch.tensor(arg.sentence_embeddings))
        c0 = arg.excerpt_indices[0]
        c1 = arg.excerpt_indices[1]
        cdc = float(sum(sim_mat[c0]) + sum(sim_mat[c1]))

        tokens = list(map(tokenize, arg.excerpt))
        texts = []
        for t in tokens:
            if len(t) > 510:
                texts.append(" ".join(t[:510]))
                log.warning(f'Shortened {arg.arg_id}\'s excerpt. Cut-off: {" ".join(t[510:])}')
            else:
                texts.append(" ".join(t))
        try:
            argumentativeness = np.mean([a['score'] for a in pipeline(texts, device=DEVICE)])
        except:
            log.error(f'Could not score argumentativeness for {arg.arg_id}.')
            argumentativeness = -1
        records.append({
            'lambda': lambda_,
            'arg_id': arg.arg_id,
            'argumentativeness': float(argumentativeness),
            'weighted_degree_centrality': cdc,
            'arg_length': len(arg.sentences),  # to normalize weighted_degree_centrality
            'soc': arg.soc_ex,
        })

    global arg_level_results
    arg_level_results = arg_level_results.append(records, ignore_index=True)


def baseline_scores(X: List[FeaturedArgument]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ec = EdgeCorrelation(f'results/{exp_id}-initial-arguments-edge-correlation.json')
    corr_original = ec.edge_correlation(X)
    log.info(f'Edge correlation of arguments as given (original): {corr_original}.')
    sc = SilhouetteCoefficient(f'results/{exp_id}-initial-arguments-silhouette-coefficient.json')
    silh_coef = sc.silhouette_coefficient(X)
    log.info(f'Silhouette coefficient of arguments as given (original): {silh_coef}.')
    current_results = {}
    # Saving results of initial arguments
    correlations = list()
    coefficients = list()
    for context in DataHandler.get_query_context_keys(X):
        current_results[f'{context}_edge_corr'] = corr_original[context].correlation
        correlations.append(corr_original[context].correlation)
        current_results[f'{context}_silhouette_coef'] = silh_coef[context]
        coefficients.append(silh_coef[context])
    current_results['avg_edge_corr'] = np.mean(np.array(correlations))
    current_results['avg_silhouette_coef'] = np.mean(np.array(coefficients))
    # endregion

    cols = ['lambda', 'avg_edge_corr', 'avg_silhouette_coef']
    for context in DataHandler.get_query_context_keys(X):
        cols.append(f'{context}_edge_corr')
        cols.append(f'{context}_silhouette_coef')

    results_tmp = pd.DataFrame(columns=cols)
    results_tmp = results_tmp.append(current_results, ignore_index=True)

    cols = ['lambda', 'arg_id', 'argumentativeness', 'weighted_degree_centrality', 'soc']
    arg_level_results_tmp = pd.DataFrame(columns=cols)

    return results_tmp, arg_level_results_tmp


def get_data(path) -> List[FeaturedArgument]:
    data = DataHandler()
    data.load_bin(path)
    X = data.get_filtered_arguments([DataHandler.get_args_filter_length(), DataHandler.get_args_filter_context_size()])

    arguments = list(map(argument_to_featured_argument, X))
    return arguments


def argument_to_featured_argument(argument: Argument) -> FeaturedArgument:
    fa = FeaturedArgument()
    fa.arg_id = argument.arg_id
    fa.sentences = argument.sentences
    fa.sentence_embeddings = argument.sentence_embeddings
    fa.query = argument.query
    fa.length = len(argument.sentences)
    fa.position = np.ones(fa.length)
    fa.word_count = np.ones(fa.length)
    fa.noun_count = np.ones(fa.length)
    fa.tfisf = np.ones(fa.length)
    fa.number_of_surface_features = 5
    fa.lr = np.ones(fa.length)
    fa.surface_features = fa._make_surface_feature_vectors()
    fa.snippet = argument.snippet
    fa.excerpt = argument.excerpt
    fa.excerpt_indices = argument.excerpt_indices
    fa.argument_embedding = argument.argument_embedding

    return fa


if __name__ == '__main__':
    main()
