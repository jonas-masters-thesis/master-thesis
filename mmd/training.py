bimport logging
import os
import pickle
import random
import time
from itertools import product
from typing import List, Union, Dict

import numpy as np
import pandas as pd

from FeaturedArgument import FeaturedArgument
from MMDBase import MMDBase
from Trainer import Trainer
from myutils import make_logging

log = logging.getLogger(__name__)
cp_dir = '.checkpoint'
cp_file = 'training.checkpoint'

# Defines whether contextualized optimization should be performed (not for cv)
CONTEXT_OPTIM = False

# Defines whether training should continue from last checkpoint. If true, ensure there is a checkpoint.
CONTINUE_FROM_CHECKPOINT = False

# Contexts that somehow cause the optimizer to set all parameters to nan
BLACKLIST = {
    'Everyone should learn to drive manual transmission cars at some point in life.',
    'This house believes that Sponge Bob Square Pants should lend his pants to Donald Duck',
    'Liberals should become more optimistic about the opportunities for success in the U.S.'
}
BLACKLIST_PATH = 'context_blacklist.txt'
TIMESTAMP = time.time()
RESULT_CSV_PATH = f'results/grid_results_cv_contextualized_optimization={CONTEXT_OPTIM}-{TIMESTAMP}.csv'


def main():
    """
    Run :code:`CONTEXT_OPTIM=True` first to find harmful contexts and set them to blacklist.
    Then run :code:`CONTEXT_OPTIM=False` to optimize hyper-parameter via cross-validation.
    """
    make_logging('training', level=logging.INFO)
    data = get_data(normalize_surface_features=True)
    training_data = make_training_data(data)

    if not CONTEXT_OPTIM:
        folds = 5
        cv_data = split_data(training_data, cv=folds)

    grid_rows = list()
    lambdas = [.125, .25, .375, .5, .625, .75, .875]
    gammas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    betas = [0.01, 0.02, 0.04, 0.08, 0.16]
    counter = 0
    combos = list(product(lambdas, gammas, betas))
    total = len(combos)

    if CONTINUE_FROM_CHECKPOINT:
        last_index = restore_from_checkpoint()
        combos = combos[last_index + 1:]
        total = len(combos)

    for lambda_, gamma, beta in combos:
        log.info(f'Running param combo {counter} of {total}')
        counter += 1
        if CONTEXT_OPTIM:
            for c in training_data.keys():
                d = training_data[c]
                trainer = Trainer(param_lambda=lambda_, param_gamma=gamma, param_beta=beta, epochs=2)
                model, losses = trainer.train(d, plot_losses=True, context=c)
                grid_rows.append({
                    'lambda': lambda_,
                    'gamma': gamma,
                    'beta': beta,
                    'loss': sum(losses),
                    'model': list(model),
                    'losses': list(losses),
                    'context': c
                })
                # trainer.create_checkpoint()
                checkpoint(combos, counter - 1)
                grid_results = save_scores(grid_rows)
        else:
            val_losses = list()
            train_losses = list()
            for i in range(folds):
                val = cv_data[i]
                train_idx = [j for j in range(folds) if j != i]
                train = list()
                for j in train_idx:
                    train.extend(cv_data[j])

                trainer = Trainer(param_lambda=lambda_, param_gamma=gamma, param_beta=beta, epochs=100)
                _, losses = trainer.train(train, plot_losses=True)
                train_losses.append(np.mean(losses))
                loss = trainer.validate(val)
                val_losses.append(loss)

            mean_train_losses = np.mean(train_losses)
            std_train_losses = np.std(train_losses)
            mean_val_losses = np.mean(val_losses)
            std_val_losses = np.std(val_losses)
            log.info(
                f'Mean train loss: {mean_train_losses}(±{std_val_losses}), mean validation loss: {mean_val_losses}(±'
                f'{std_train_losses})')
            grid_rows.append({
                'lambda': lambda_,
                'gamma': gamma,
                'beta': beta,
                'mean_val_losses': mean_val_losses,
                'std_val_losses': std_val_losses,
                'mean_train_losses': mean_train_losses,
                'std_train_losses': std_train_losses,
            })
            checkpoint(combos, counter - 1)
            grid_results = save_scores(grid_rows)

    if not CONTEXT_OPTIM:
        min_idx = grid_results['mean_val_losses'].idxmin()
        min_params = grid_results.iloc[min_idx]
        refit(min_params['lambda'], min_params['gamma'], min_params['beta'], training_data)


def save_scores(grid_rows):
    grid_results = pd.DataFrame.from_records(grid_rows)
    grid_results.to_csv(RESULT_CSV_PATH, index=False, header=True)
    return grid_rows


def refit(lambda_: float, gamma: float, beta: float, data: List[MMDBase.triplet]):
    log.info(f'Refit model with lambda={lambda_}, gamma={gamma}, beta={beta}.')
    trainer = Trainer(param_lambda=lambda_, param_gamma=gamma, param_beta=beta, epochs=100)
    model, losses = trainer.train(data, plot_losses=True, context='')
    trainer.create_checkpoint()
    log.info(f'Final model parameters: {model}')
    with open('results/SupMMD-Model.pickle', 'wb') as file:
        pickle.dump(obj=model, file=file)
    pd.DataFrame.from_records([{'epoch': idx + 1, 'loss': loss} for idx, loss in enumerate(losses)]).to_csv(
        'results/SupMMD-losses.csv', header=True, index=False)


def checkpoint(combos, combo_idx):
    lambda_, gamma, beta = combos[combo_idx]
    cp = {
        'lambda_': lambda_,
        'gamma': gamma,
        'beta': beta,
        'index': combo_idx
    }
    if not os.path.isdir(cp_dir):
        os.mkdir(cp_dir)
    with open(f'{cp_dir}/{cp_file}', 'wb') as file:
        pickle.dump(cp, file)


def restore_from_checkpoint(path=f'{cp_dir}/{cp_file}') -> int:
    """
    Loads checkpoint from file
    :param path: path of the checkpoint
    :return: index of last param combo
    """
    with open(path, 'rb') as file:
        cp = pickle.load(file)

    return cp['index']


def split_data(data: List[MMDBase.triplet], cv: int):
    """
    Splits the data for cross-validation
    :param data: training data to be splitted
    :param cv: number of folds
    :return: list of list of instance, each outer list is one fold
    """
    # n = len(data)
    # n_per_fold = n // cv
    cv_data = [list() for _ in range(cv)]
    for i, tri in enumerate(data):
        idx = i % cv
        cv_data[idx].append(tri)

    return cv_data


def make_training_data(data: List[FeaturedArgument]) -> Union[List[MMDBase.triplet], Dict[str, List[MMDBase.triplet]]]:
    """
    Depending on :code:`CONTEXT_OPTIM` the function returns either a dict of list of triplets per context Dict[str,
    List[Trainer.triplet]], or a list of triplets List[Trainer.triplets]
    """
    contexts_keys = set([a.query for a in data])
    keys = contexts_keys - BLACKLIST
    contexts = {c: [] for c in keys}
    for a in data:
        if a.query in contexts.keys():
            contexts[a.query].append(a)

    log.info(f'Number of contexts: {len(contexts)}')
    keys = random.choices(list(contexts.keys()), k=20)
    if CONTEXT_OPTIM:
        triplets = dict()
        for c in keys:
            context = contexts[c]
            if len(context) > 1:
                for focal_arg in context:
                    tri = make_triplet(focal_arg, context)
                    if c not in triplets.keys():
                        triplets[c] = list()
                    triplets[c].append(tri)
            else:
                log.error(f'Excluded context due to too few arguments ({len(context)}): "{c}"')
    else:
        triplets = list()
        for c in keys:
            context = contexts[c]
            if len(context) > 1:
                for focal_arg in context:
                    triplets.append(make_triplet(focal_arg, context))
            else:
                log.error(f'Excluded context due to too few arguments ({len(context)}): "{c}"')
    return triplets


def make_triplet(argument, context) -> MMDBase.triplet:
    context_wo_focal = [arg for arg in context if arg != argument]
    return MMDBase.triplet(argument, FeaturedArgument.context_argument_dummy(context_wo_focal),
                           FeaturedArgument.only_snippet_argument_dummy(argument))


def normalize(data: List[FeaturedArgument]):
    # find maximums
    max_values = data[0].surface_features[0]  # features of the first sentence of the first argument
    expected_number_of_sf = len(max_values)
    for argument in data:
        for sfs in argument.surface_features:  # iterate over sentences' surface features
            if expected_number_of_sf != len(sfs):
                raise ValueError(f'Length mismatch: expected: {expected_number_of_sf}, actual: {len(sfs)}.')
            for i in range(expected_number_of_sf):
                if sfs[i] > max_values[i]:
                    max_values[i] = sfs[i]

    # normalize
    for argument in data:
        argument.surface_features = argument.surface_features / max_values


def get_data(normalize_surface_features: bool = False) -> List[FeaturedArgument]:
    with open(
            '../heuristic-data-creation/data/GW '
            '2021-08-31/FeaturedArguments-w-reference-snippets-r_0.1-args-me-supmmd-train.pickle',
            'rb') as f:
        X = pickle.load(f)  # [:10000]

    embs = np.load('embeddings.npa.npy', allow_pickle=True)
    for i in range(len(X)):
        X[i].sentence_embeddings = embs[i]
        X[i].query = X[i].topic

    if normalize_surface_features:
        normalize(X)

    log.debug(f'Loaded {len(X)} arguments.')
    return X


def keep_argument(argument: FeaturedArgument):
    if argument.sentence_embeddings is None or len(argument.sentence_embeddings) == 0 \
            or argument.sentences is None or len(argument.sentences) == 0 \
            or argument.surface_features is None or len(argument.surface_features) == 0 \
            or argument.query is None or len(argument.query) == 0 \
            :
        return False
    return True


def update_blacklist():
    with open(BLACKLIST_PATH, 'r', encoding='utf-8') as f:
        BLACKLIST.update([s.rstrip() for s in f.readlines()])


if __name__ == '__main__':
    update_blacklist()
    main()
