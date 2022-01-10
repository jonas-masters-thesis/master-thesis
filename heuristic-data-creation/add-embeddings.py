import pickle

from WordEmbeddingTransformer import WordEmbeddingTransformer
from myutils import make_logging


def main():
    make_logging('add_embeddings')
    with open(
            'data/1632239915.4824035-3756-arguments-cleaned-test-w-features.pickle', 'rb') as f:
        X = pickle.load(f)

    emb = WordEmbeddingTransformer()

    upper_bound = len(X) // 10000

    for i in range(upper_bound + 1):
        if (i + 1) * 10000 >= len(X):
            emb.transform(X[i * 10000:])
            with open(
                    'data/GW '
                    '2021-08-31/FeaturedArguments-w-reference-snippets-r_0.1-args-me-supmmd-train-w-embeddings.pickle',
                    'wb') as f:
                pickle.dump(X, f)
        else:
            emb.transform(X[i * 10000:(i + 1) * 10000])
            with open(
                    'data/GW '
                    f'2021-08-31/FeaturedArguments-w-reference-snippets-r_0.1-args-me-supmmd-train-[{i * 10000}:'
                    f'{(i + 1) * 10000}].pickle',
                    'wb') as f:
                pickle.dump(X, f)

    # data = DataHandler()
    # data.load_bin('../../not-gitted/argsme-crawled/1629700068.9873986-6578-arguments.pickle')
    # id_arg_map: Dict[str, Argument] = dict()
    # for argument in data.get_arguments():
    #     id_arg_map[argument.arg_id] = argument
    #
    # del data
    #
    # for a in X:
    #     a.sentence_embeddings = id_arg_map[a.arg_id].sentence_embeddings

    pass


if __name__ == '__main__':
    main()
