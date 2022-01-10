import json
import logging
import pickle
import time
from typing import List, Dict, Tuple

import requests
from nltk import sent_tokenize

from Argument import Argument
from WordEmbeddingTransformer import WordEmbeddingTransformer
from myutils import make_logging

log = logging.getLogger(__name__)
BUILD_EMBEDDINGS = True
BASE_PATH = '../data/'


def main():
    make_logging('argsme-crawler')
    queries = get_queries()
    arguments: List[Argument] = list()
    arguments_json = list()
    stances: Dict[str, str] = dict()
    for q in queries:
        try:
            response = api_query(q)
            arguments_json.append(response)
            args, stncs = parse_json(response)
            arguments.extend(args)
            stances.update(stncs)
        except:
            log.error(f'Could not process {q}.')

    if BUILD_EMBEDDINGS:
        embedder = WordEmbeddingTransformer()
        arguments = embedder.transform(arguments)

    name_id = time.time()
    with open(f'{BASE_PATH}{name_id}-{len(arguments)}-arguments.stance', 'w') as f:
        json.dump(stances, f, indent=4)
    with open(f'{BASE_PATH}{name_id}-{len(arguments)}-arguments.pickle', 'wb') as f:
        pickle.dump(arguments, f)

    log.info(f'Crawled {len(arguments)} arguments.')


def get_queries():
    with open('wiki_controversies.txt', 'r') as file:
        lines = file.readlines()
    return [l.strip() for l in lines]


def encode(query: str):
    if ' ' in query:
        query = query.replace(' ', '+')
        #query = '"' + query + '"'
    return query


def api_query(query: str):
    query = encode(query)
    log.debug(f'Querying {query} ...')
    response = requests.get(f'https://www.args.me/api/v2/arguments?query={query}&format=json')
    j_response = json.loads(response.text)
    if 'arguments' not in j_response or len(j_response['arguments']) == 0:
        log.warning(f'No arguments returned for {query}.')
    return j_response


def parse_json(body) -> Tuple[List[Argument], Dict[str, str]]:
    arguments: List[Argument] = list()
    stances: Dict[str, str] = dict()
    if 'arguments' in body:
        for json_arg in body['arguments']:
            a = Argument()
            a.arg_id = json_arg['id']
            a.query = body['query']['text'].replace(' ', '_').replace('"', '')
            a.sentences = sent_tokenize(" ".join([p['text'] for p in json_arg['premises']]))
            stances[a.arg_id] = json_arg['stance']
            arguments.append(a)

    return arguments, stances


if __name__ == '__main__':
    main()
