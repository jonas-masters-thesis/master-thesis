import json
import logging
import pickle

import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from myutils import make_logging

log = logging.getLogger(__name__)


# TODO: Run for args-me too
def main():
    make_logging('argQ-scoring')
    log.info(torch.__version__)
    log.info(torch.cuda.is_available())
    log.info(torch.cuda.device_count())
    if torch.cuda.device_count() >= 1:
        DEVICE = torch.cuda.current_device()
        log.info(torch.cuda.current_device())
        log.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        DEVICE = -1  # use cpu
    # https://huggingface.co/transformers/main_classes/pipelines.html#textclassificationpipeline

    model = BertForSequenceClassification.from_pretrained('../bert-finetuning/results/argQ-bert-base-uncased',
                                                          local_files_only=True,
                                                          cache_dir='cache')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', task='ArgQ')

    file_base_name = '1629700068.9873986-6578-arguments'
    with open(f'../../not-gitted/argsme-crawled/{file_base_name}.pickle', 'rb') as f:
        data = pickle.load(f)

    arg_sent_Q = dict()
    for argument in tqdm(data):
        try:
            results = pipeline(argument.sentences, device=DEVICE)
            arg_sent_Q[argument.arg_id] = [float(r['score']) for r in results]
        except:
            arg_sent_Q[argument.arg_id] = []

    json.dump(arg_sent_Q, open(f'results/{file_base_name}-sent-argQ.json', 'w'), indent=4)


if __name__ == '__main__':
    main()
