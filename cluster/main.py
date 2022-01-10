import logging
from datetime import date
from sklearn.pipeline import Pipeline
import sys

from CohesionScorer import CohesionScorer
from SeparationScorer import SeparationScorer
from SilhouetteScorer import SilhouetteScorer
from shared.DataHandler import DataHandler


def main():
    # region logging
    # https://appdividend.com/2019/06/08/python-logging-tutorial-with-example-logging-in-python/
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log = logging.getLogger()
    level = logging.DEBUG
    log.setLevel(level)
    shandler = logging.StreamHandler(sys.stdout)
    fhandler = logging.FileHandler(f'logs/silhouette-{date.today()}.log')
    shandler.setLevel(level)
    fhandler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(name)s \t [%(levelname)s] \t %(message)s')
    shandler.setFormatter(formatter)
    fhandler.setFormatter(formatter)
    log.addHandler(shandler)
    log.addHandler(fhandler)
    # endregion

    data = DataHandler()
    data.load_bin('../../not-gitted/dataset_as_json_file.pickle')  # embeddings are pre-computed
    X = data.get_arguments()[:10]

    pipeline = Pipeline(steps=[
        ('cohesion', CohesionScorer()),
        ('separation', SeparationScorer()),
        ('silhouette', SilhouetteScorer())
    ])
    pipeline.predict(X)

    # compute trade-off (ssc, soc)
    # trade_off = TradeOffScorer()
    # trade_off.transform(X)

    data.save_results('results/silhouette.json')


if __name__ == '__main__':
    main()
