import json
import logging

import pandas as pd
from nltk import word_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm

from DataHandler import DataHandler
from myutils import make_logging

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)


class DatasetCreator:

    def __init__(self):
        pass

    def load_data(self, path):
        data = DataHandler()
        data.load_json(path)
        return data.get_arguments()


def main():
    make_logging('datacreation')
    log = logging.getLogger(__name__)

    with open('../../not-gitted/argsme-1.0-cleaned/args-me-1.0-cleaned-as-list.json', 'r') as f:
        d = json.load(f)

    filter_conclusions(d)

    quit(0)
    with open('results/args-me-1.0-cleaned-with-snippets.json', 'w', encoding='utf-8') as g:
        json.dump(d['arguments'], g)

    log.info('finish')


def filter_conclusions(arguments):
    rows = list()
    for argument in tqdm(arguments):
        conclusion = argument['conclusion']
        id = argument['id']
        conc_len = len(word_tokenize(conclusion))
        rows.append({'id': id, 'conclusion': conclusion, 'conc_len': conc_len})

    rows = list({v['id'][:-9]: v for v in rows}.values())
    args = pd.DataFrame.from_records(rows)
    args.to_csv('results/conclusions.csv', index=False, header=True)


if __name__ == '__main__':
    main()
