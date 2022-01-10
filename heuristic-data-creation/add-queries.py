import json
import logging
import pickle

from tqdm import tqdm

from myutils import make_logging

log = logging.getLogger(__name__)
PATH = 'data/GW 2021-08-31/FeaturedArguments-w-reference-snippets-r_0.1-args-me-supmmd-train-[0%3A10000].pickle'


def main():
    make_logging('dummy')
    with open(PATH, 'rb') as f:
        args = pickle.load(f)

    with open('data/filtered-w-reference-snippets-r_0.1-args-me.json', 'r') as f:
        d = json.load(f)

    d_dict = {a['id']: a['conclusion'] for a in d}

    for argument in tqdm(args[:10000]):
        argument.query = d_dict[argument.arg_id]

    with open(PATH, 'wb') as f:
        pickle.dump(args, f)


if __name__ == '__main__':
    main()
