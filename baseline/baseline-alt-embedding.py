import json
import logging
from typing import List

from nltk import sent_tokenize

from Argument import Argument
from DataHandler import DataHandler
from WordEmbeddingTransformer import WordEmbeddingTransformer
from myutils import make_logging

log = logging.getLogger(__name__ + 'baseline-alt-embedding')


def main():
    make_logging('baseline-alt-embedding')

    data = read_data('../argsme-api/results/full-text-test-corpus.json',
                     '../contra-lexrank/results/GW 2021-10-05 Test/arg_contexts.json')
    embedder = WordEmbeddingTransformer()
    embedder.transform(data)

    handler = DataHandler(data)
    handler.dump_data('test-data-unprocessed.pickle')


def read_data(fulltext_path, argcontext_path) -> List[Argument]:
    arguments = json.load(open(fulltext_path))
    contexts = json.load(open(argcontext_path))

    for argument in arguments:
        if argument['arg_id'] in contexts.keys():
            argument['query'] = contexts[argument['arg_id']]
            argument['sentences'] = sent_tokenize(argument['text'])

        else:
            log.warning(f'no query for {argument["arg_id"]}')

    arguments = list(map(parse_json, arguments))

    return arguments


def parse_json(j) -> Argument:
    return Argument(sentences=j['sentences'],
                    arg_id=j['arg_id'],
                    query=j['query'])


if __name__ == '__main__':
    main()
