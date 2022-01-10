from unittest import TestCase

from shared.ArgumentativenessScorer import ArgumentativenessScorer

from shared.DataHandler import DataHandler


class TestArgumentativenessScorer(TestCase):

    def test_transform(self):
        data = DataHandler()
        data.load_json('../../not-gitted/dataset_as_json_file.json')
        X = data.get_arguments()[:10]

        scorer = ArgumentativenessScorer()
        result = scorer.transform(X)
        for argument in result:
            self.assertEqual(len(argument.sentences), len(argument.argumentativeness_scores))

    def test_compute_arg_vector(self):
        data = DataHandler()
        data.load_json('../../not-gitted/dataset_as_json_file.json')
        X = [data.get_arguments()[0], data.get_arguments()[3]]

        scorer = ArgumentativenessScorer()
        scorer.transform(X)
        for arg in X:
            self.assertEqual(len(arg.sentences), len(arg.argumentativeness_scores))
            self.assertTrue(sum(arg.argumentativeness_scores) > 0)
