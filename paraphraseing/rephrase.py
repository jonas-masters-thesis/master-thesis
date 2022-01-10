import argparse
import json
import logging
from collections import namedtuple
from typing import List

import torch
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from myutils import make_logging

ArgSnip = namedtuple('ArgSnip',
                     ['arg_id',
                      'snippet_id',
                      'query',
                      'sentences',
                      'snippet',
                      'paraphrase_1',
                      'paraphrase_2',
                      'paraphrase_both'])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log = logging.getLogger(f'{__name__}-rephrase')

model_name = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(DEVICE)

ORIGIN_FILE = '../baseline/results/baseline-test-predictions-w-snippetId.json'
TARGET_FILE = '../baseline/results/baseline-test-predictions-w-snippetId.json'


def main():
    make_logging('rephrase', level=logging.INFO)
    log.debug(f'Paraphrasing {ORIGIN_FILE} to {TARGET_FILE}')

    arguments = get_data(ORIGIN_FILE)
    paraphrased_arguments = list()
    for argument in tqdm(arguments):
        res_1 = get_response(argument.snippet[0], 3, 3, argument.arg_id)
        res_2 = get_response(argument.snippet[1], 3, 3, argument.arg_id)
        res_both = get_response(" ".join(argument.snippet), 3, 3, argument.arg_id)
        paraphrased_arguments.append(
            ArgSnip(argument.arg_id, argument.snippet_id, argument.query, argument.sentences, argument.snippet, res_1,
                    res_2, res_both)
        )

    dump(paraphrased_arguments)


def get_response(input_text, num_return_sequences, num_beams, arg_id) -> List[str]:
    try:
        log.debug(f'Paraphrasing: {input_text}')
        batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
            DEVICE)
        translated = model.generate(**batch,
                                    max_length=60,
                                    num_beams=num_beams,
                                    num_return_sequences=num_return_sequences,
                                    temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        log.debug(f'Paraphrased: {tgt_text}')
        return tgt_text
    except Exception as e:
        log.error(f'Transforming the snippet failed for argument {arg_id}: {e}')
        return []


def dump(arguments: List[ArgSnip]):
    def parse(a: ArgSnip):
        return {
            'query': a.query,
            'arg_id': a.arg_id,
            'snippet_id': a.snippet_id,
            'sentences': a.sentences,
            'excerpt': a.snippet,
            'paraphrased_1': a.paraphrase_1,
            'paraphrased_2': a.paraphrase_2,
            'paraphrased_both': a.paraphrase_both,
        }

    arguments_to_dump = list(map(parse, arguments))
    with open(TARGET_FILE, 'w') as file:
        json.dump(arguments_to_dump, file, indent=4)
    log.debug(f'Dumped arguments to {TARGET_FILE}')


def get_data(path) -> List[ArgSnip]:
    with open(path, 'r', encoding='utf-8') as file:
        raw = json.load(file)

    arguments = list()
    for a in raw:
        arguments.append(
            ArgSnip(a['arg_id'], a['snippet_id'], a['query'], a['sentences'], a['excerpt'], '', '', '')
        )
    log.debug(f'Loaded {len(arguments)} arguments.')
    return arguments


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-i', type=str)
    # parser.add_argument('-o', type=str)
    #
    # args = parser.parse_args()
    # ORIGIN_FILE = args.i
    # TARGET_FILE = args.o

    main()
