import pickle
from unittest import TestCase

from tqdm import tqdm

from DataHandler import DataHandler
from SBERTHandler import SBERTHandler
from WordEmbeddingTransformer import WordEmbeddingTransformer


class TestDataEmbedding(TestCase):

    def test_transform(self):
        data = DataHandler()
        data.load_json('../../not-gitted/dataset_as_json_file.json')
        self.assertEqual(len(data.get_arguments()), 100)

        embedder = WordEmbeddingTransformer()
        result = embedder.transform(data.get_arguments())

        for argument in result:
            self.assertEqual(len(argument.sentences), len(argument.sentence_embeddings))
            self.assertFalse(len(argument.argument_embedding) == 0)

    def test_SaveTransform(self):
        data = DataHandler()
        data.load_json('../../not-gitted/dataset_as_json_file.json')
        print(len(data.get_arguments()))

        embedder = WordEmbeddingTransformer()
        embedder.transform(data.get_arguments())

        data.dump_data('../../not-gitted/dataset_as_json_file.pickle')

    def test_SaveTransform_ArgsMe(self):
        data = DataHandler()
        data.load_argsme_json('../heuristic-data-creation/data/filtered-args-me.json')
        print(len(data.get_arguments()))

        embedder = WordEmbeddingTransformer()
        embedder.transform(data.get_arguments())

        data.dump_data('../../not-gitted/filtered-args-me.json.pickle')

    def test_SaveTransform_ArgsMe_w_Features(self):
        with open('../heuristic-data-creation/data/features.json.pickle', 'rb') as f:
            d = pickle.load(f)

        handler = SBERTHandler()

        for argument in tqdm(d):
            argument['sentence_embeddings'] = handler.sentence_vector(argument['premises'][0]['sentences'])

        with open('../../not-gitted/filtered-w-reference-snippets-a-features-r_0.1-args-me.json.pickle',
                  'wb') as f:
            pickle.dump(d[:100], f)

    def test_loadDump(self):
        data = DataHandler()
        data.load_bin('../../not-gitted/dataset_as_json_file.pickle')
        self.assertEqual(len(data.get_arguments()), 100)
