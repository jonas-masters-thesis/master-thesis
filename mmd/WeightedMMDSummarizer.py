import logging
from typing import List

from Argument import Argument
from MMDSummarizer import MMDSummarizer


class WeightedMMDSummarizer(MMDSummarizer):

    def __init__(self, param_gamma, param_lambda, snippet_length):
        super(WeightedMMDSummarizer, self).__init__(param_gamma, param_lambda, snippet_length)
        self.log = logging.getLogger(WeightedMMDSummarizer.__name__)

    def fit_context(self, arguments: List[Argument]):
        self.log.info(f'Test override. {len(arguments)} arguments on context {arguments[0].query}')

    def squared_mmd(self, X, Y):
        """
        Computes weighted squared MMD for given input.

            :math:`MMD^2(X,Y) = \\frac{1}{n^2} \\sum_{i=1}^n \\sum_{j=1}^n f_\\theta^x(x_i) \\cdot f_\\theta^x(x_j)
            \\cdot k(x_i, x_j)
            - \\frac{2}{nm} \\sum_{i=1}^n \\sum_{j=1}^m f_\\theta^x(x_i) \\cdot f_\\theta^y(y_j) \\cdot k(x_i, y_j)
            + \\frac{1}{m^2} \\sum_{i=1}^m \\sum_{j=1}^m f_\\theta^y(y_i) \\cdot f_\\theta^y(y_j) \\cdot k(y_i, y_j)`
        """
        result = 0
        n = len(X)
        m = len(Y)
        if n == 0 or m == 0:
            return 0

        for x_i in X:
            for x_j in X:
                result += (1. / (n * n)) * self.f_x(x_i) * self.f_x(x_j) * self.kernel(x_i, x_j, self.param_gamma)

        for x_i in X:
            for y_j in Y:
                result += - (2. / (n * m)) * self.f_x(x_i) * self.f_y(y_j) * self.kernel(x_i, y_j, self.param_gamma)

        for y_i in Y:
            for y_j in Y:
                result += (1. / (m * m)) * self.f_y(y_i) * self.f_y(y_j) * self.kernel(y_i, y_j, self.param_gamma)

        return result

    def f_x(self, x):
        return 1

    def f_y(self, y):
        return 1
