import logging

import torch

from Inference import Inference
from myutils import make_logging
from training import get_data

log = logging.getLogger(__name__)


def main():
    make_logging('mmd')

    data = get_data(normalize_surface_features=True)

    thetaAB = torch.tensor(
        [-1.4727394169625077, -0.5994600705543802, -0.48164147188298384, -0.4363084702811584, 0.1536314497528065,
         0.032945744025799806], dtype=torch.float64)

    summarizer = Inference(param_gamma=.5, param_lambda=.375, snippet_length=2, thetaA=thetaAB, thetaB=thetaAB)
    summarizer.fit(data)


if __name__ == '__main__':
    main()
