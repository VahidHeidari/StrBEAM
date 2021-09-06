import copy
import unittest

import numpy as np

import cal_gamma
import calc_alpha_beta
import common
import data_set_5
import disease_graph



pA = 0.1
ALPHA = 1.5

EXP_DGAMMA0  = [  34.5844,   42.0357,     55.4119,   52.903,    48.1057 ]
EXP_DGAMMA1  = [  -2.36896,  -0.0729028,   6.92428,   9.67806,  10.3103 ]
EXP_DGAMMA02 = [ -40.2167,  -40.2167,    -27.3965,  -26.6381,  -25.655  ]
EXP_DGAMMA12 = [ -38.0928,  -37.5623,    -29.6991,  -28.9406,  -27.9576 ]
EXP_DGAMMAS = [ EXP_DGAMMA0, EXP_DGAMMA1, EXP_DGAMMA02, EXP_DGAMMA12 ]


#
# dgamma format:
#
#   [ c1, c2, c3,   c4, c5,   ... ]
#     \---v---/     \--v--/
#       comp1        comp2
#
# Cliques in components are consecutive.
#
def GetDLP(dataC, dataU, graph, marker, mAlpha, mBeta):
    modules = graph.GetComponents()
    num_nodes = np.sum([len(m) for m in modules])
    module_idx, marker_idx = graph.FindMarker(marker)

    dgamma0 = np.zeros((num_nodes), dtype=float) if num_nodes > 0 else []
    dgamma1 = np.zeros((num_nodes), dtype=float) if num_nodes > 0 else []
    dgamma02 = np.zeros((num_nodes), dtype=float) if num_nodes > 0 else []
    dgamma12 = np.zeros((num_nodes), dtype=float) if num_nodes > 0 else []

    i = 0
    for component in modules:
        for clique_idx in component:
            g1 = graph.GetCliqueCopy(clique_idx)                                # Exclude marker if it is in clique_idx.
            if clique_idx == module_idx:
                del g1[marker_idx]

            c1 = [ marker ]
            dgamma0[i] = cal_gamma.CalGamma(dataC, dataU, 1.0, 1.0, g1, c1, 0)  # rt_cu
            dgamma1[i] = cal_gamma.CalGamma(dataC, dataU, 1.0,  pA, g1, c1, 1)  # rt_c + _mixLP(rt_u, pA)

            dgamma02[i] = dgamma0[i] + mAlpha[marker]
            dgamma12[i] = dgamma1[i] + mBeta[marker]

            for k in range(graph.GetNumCliques()):                              # Iterate connections of c_i
                if not graph.HasInteraction(i, k):                              # If i == k then HasInteractions returns false, so it contiues when i == k.
                    continue

                g2 = graph.GetCliqueCopy(k)
                if k == module_idx:
                    del g2[marker_idx]
                    if len(g2) < 1:
                        continue

                dgamma02[i] += cal_gamma.CalGamma(dataC, dataU, 1.0, pA, g1 + [marker], g2, 0)
                dgamma02[i] -= cal_gamma.CalGamma(dataC, dataU, 1.0, pA, g1,            g2, 0)
                dgamma12[i] += cal_gamma.CalGamma(dataC, dataU, 1.0, pA, g1 + [marker], g2, 1)
                dgamma12[i] -= cal_gamma.CalGamma(dataC, dataU, 1.0, pA, g1,            g2, 1)

            i += 1                                                              # Next node in graph
    return dgamma0, dgamma1, dgamma02, dgamma12



#
# components: list of connected components, i.e. { w_1, ..., w_W }, where
# w_i = { c_1, ..., c_W_i } are connected cliques. For instance, the following
# graph makes components like components = [ [ n0, n1 ], [ n2, n3, n4 ] ].
#
#         .------.       .------.
#        /   n0   \____ /   n1   \
#        \ {0, 1} /     \ {2, 3} /
#         '------'       '------'
#
#      .---------.       .--------.
#     /     n2    \____ /    n3    \
#     \ {5, 6, 7} /     \ {10, 13} /
#      '---------'       '--------'
#              \
#               \ .----.
#                /  n4  \
#                \ {14} /
#                 '----'
#
# Connected components (modules):
#     { n0, n1 }
#     { n2, n3, n4 }
#
def InitializeGraph():
    graph = disease_graph.Graph()
    graph.AddClique([0, 1])
    graph.AddClique([2, 3])
    graph.AddClique([5, 6, 7])
    graph.AddClique([10, 13])
    graph.AddClique([14])
    graph.SetInteraction(0, 1)
    graph.SetInteraction(2, 3)
    graph.SetInteraction(2, 4)
    #print(graph.ToString())
    return graph


class GetDLPTestCase(unittest.TestCase):
    def CheckDgammaEq(self, dgammas, exp_dgammas):
        if len(dgammas) != len(exp_dgammas):
            print('The length of dgammas({}) and exp_dgammas({}) are not equal!'.format(len(dgammas), len(exp_dgammas)))
        self.assertEqual(len(dgammas), len(exp_dgammas))

        for i in range(len(dgammas)):
            if len(dgammas[i]) != len(exp_dgammas[i]):
                print('The length of dgammas[i:{}]({}) and exp_dgammas[i:{}]({}) are not equal!'.format(i, len(dgammas), i, len(exp_dgammas)))
            self.assertEqual(len(dgammas[i]), len(exp_dgammas[i]))

            for j in range(len(dgammas[i])):
                if dgammas[i][j] != exp_dgammas[i][j]:
                    if abs(dgammas[i][j] - exp_dgammas[i][j]) > 1e-4:
                        print('dgammas[i:{}][j:{}]={} is not equal to exp_dgammas[i:{}][j:{}]={}'.format(i, j, dgammas[i][j], i, j, exp_dgammas[i][j]))
                self.assertFalse(abs(dgammas[i][j] - exp_dgammas[i][j]) > 1e-4)
        print('OK!    dgammas == exp_dgammas')


    def test_GetDLP(self):
        graph = InitializeGraph()
        num_loci = len(data_set_5.dataC[0])
        dataU = data_set_5.dataU
        dataC = data_set_5.dataC
        counts = [ [ common.CountCombinations(dataU, [i]),
                    common.CountCombinations(dataC, [i]) ] for i in range(num_loci) ]
        mAlpha, mBeta = calc_alpha_beta.CalcAlphaBeta(counts, ALPHA)
        dgamma0, dgamma1, dgamma02, dgamma12 = GetDLP(dataC, dataU, graph, 2, mAlpha, mBeta)
        print(dgamma0)
        print(dgamma1)
        print(dgamma02)
        print(dgamma12)
        #print(graph.ToString())
        self.CheckDgammaEq([ dgamma0, dgamma1, dgamma02, dgamma12 ], EXP_DGAMMAS)


if __name__ == '__main__':
    unittest.main()

