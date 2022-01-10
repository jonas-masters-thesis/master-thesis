from unittest import TestCase

from CentralityScorer import CentralityScorer
from shared.DataHandler import DataHandler


class TestCentralityScorer(TestCase):

    def test_transform(self):
        data = DataHandler()
        data.load_bin('../../not-gitted/dataset_as_json_file.pickle')
        X = data.get_arguments()[:10]

        scorer = CentralityScorer()
        result = scorer.transform(X)
        print(len(result))
        for argument in result:
            self.assertEqual(len(argument.sentence_embeddings), len(argument.centrality_scores))
