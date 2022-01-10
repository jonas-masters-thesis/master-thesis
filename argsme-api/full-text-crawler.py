import argparse
import json
import logging
from typing import Dict, Optional

import requests

from myutils import make_logging

log = logging.getLogger(__name__)
ORIGIN_FILE = 'test_ids.json'
TARGET_FILE = None


def main():
    make_logging('full-test-crawler')

    with open(ORIGIN_FILE, 'r') as file:
        test_ids = json.load(file)

    arguments = list()
    for i in test_ids['ids']:
        res = api_query(i)
        a = parse_json(res)
        if a is not None:
            arguments.append(a)

    with open(TARGET_FILE, 'w') as file:
        json.dump(arguments, file, indent=4)


def api_query(id: str):
    log.debug(f'Querying {id} ...')
    url = f'https://www.args.me/api/v2/arguments/{id}?format=json'
    response = requests.get(url)
    j_response = json.loads(response.text)
    if j_response is None:
        log.error(f'No argument found for {id}.')
    return j_response


def parse_json(body) -> Optional[Dict]:
    if 'id' in body and 'premises' in body:
        return {
            'arg_id': body['id'],
            'text': " ".join([p['text'] for p in body['premises']])
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-i', type=str)
    parser.add_argument('-o', type=str)

    args = parser.parse_args()
    # ORIGIN_FILE = args.i
    TARGET_FILE = args.o

    main()
