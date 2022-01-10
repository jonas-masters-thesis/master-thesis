import abc


class Heuristic(abc.ABC):

    @abc.abstractmethod
    def score_summary_candidate(self, sentence, claim):
        pass
