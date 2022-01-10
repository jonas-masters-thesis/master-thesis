from typing import Dict, Any

import datasets
from datasets import MetricInfo
from sklearn.metrics import r2_score


# https://github.com/huggingface/datasets/blob/master/metrics/recall/recall.py
# https://huggingface.co/docs/datasets/loading_metrics.html
class R2(datasets.Metric):
    def _info(self) -> MetricInfo:
        return datasets.MetricInfo(
            description="R squared",
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
                }
            ),
            reference_urls=[
                'https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination']
        )

    def _compute(self, *, predictions=None, references=None, **kwargs) -> Dict[str, Any]:
        return {
            "r2": r2_score(
                references,
                predictions,
            )
        }
