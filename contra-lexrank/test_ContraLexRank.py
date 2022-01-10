from unittest import TestCase

import numpy as np

from ContraLexRank import ContraLexRank


class TestContraLexRank(TestCase):
    def test_power_iteration(self):
        M = np.array([[0, 1], [1, 0]])
        clr = ContraLexRank(1 / 3, 1 / 3, 1 / 3, 2)
        sol = clr.power_iteration(M, 10e-8)
        print(M)
        print(sol)

    def test_power_iteration2(self):
        M = np.array([[0.5, .5], [.2, .8]])
        clr = ContraLexRank(1 / 3, 1 / 3, 1 / 3, 2)
        sol = clr.power_iteration(M, 10e-8)
        print(M)
        print(sol)
