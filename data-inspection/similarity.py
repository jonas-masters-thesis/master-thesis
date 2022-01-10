import logging

import torch
from matplotlib import pyplot as plt

from DataHandler import DataHandler
from MMDBase import MMDBase
from myutils import make_logging

log = logging.getLogger(__name__)

F_LIST = [
    'climate change-06',
    # 'climate change-07',
    'climate change-08',
]


def main():
    make_logging('similarity', level=logging.INFO)
    data = DataHandler()
    data.load_bin('../../not-gitted/dataset_as_json_file.pickle')  # embeddings are pre-computed
    log.debug(len(data.get_arguments()))
    # X = [a for a in data.get_arguments() if a.arg_id in F_LIST]
    cc_06 = [a for a in data.get_arguments() if a.arg_id in F_LIST[0]][0]
    cc_08 = [a for a in data.get_arguments() if a.arg_id in F_LIST[1]][0]
    log.info(f'{cc_06.arg_id}')

    # steal the function
    sim_func = MMDBase(param_gamma=.0, param_lambda=.0).cosine_kernel_matrix
    cc_06_tensor = torch.tensor(cc_06.sentence_embeddings)
    cc_08_tensor = torch.tensor(cc_08.sentence_embeddings)

    sim_matrix = sim_func(cc_06_tensor, cc_08_tensor)
    plot_sim_heat(sim_matrix, save='cc-06-08-heatmap.pdf')
    log.info(f'cc_06 has {len(cc_06.sentences)} sentences.')
    log.info(f'cc_08 has {len(cc_08.sentences)} sentences.')


def plot_sim_heat(similarities, save=None):
    plt.xlabel('CC-08')
    plt.ylabel('CC-06')
    ax = plt.subplot(111)
    # ax.set_xlim(1, 4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([f'{i + 1:02d}' for i in range(similarities.shape[1])])
    ax.set_yticklabels([f'{i:02d}' for i in range(similarities.shape[0] + 1)])
    plt.imshow(similarities, cmap=plt.cm.BuPu_r)
    plt.colorbar()
    if save is not None:
        plt.savefig(save)
    plt.show()


if __name__ == '__main__':
    main()
