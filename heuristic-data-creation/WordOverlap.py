from rouge_score import rouge_scorer

from Heuristic import Heuristic


class WordOverlap(Heuristic):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)

    def score_summary_candidate(self, sentence, claim):
        return self.scorer.score(sentence, claim)
