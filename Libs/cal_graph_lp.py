import unittest

import cal_full_lp
import cal_gamma
import calc_alpha_beta
import data_set_4
import disease_graph
import priors



def CalGraphLP(dataC, dataU, mmember, graph, mAlpha, mBeta):
    lp0 = lp1 = 0.0
    for i in range(graph.GetNumCliques()):
        for m in graph.GetMarkers(i):
            lp0 += mAlpha[m]
            lp1 += mBeta[m]
        if graph.GetNumMarkers(i) > 1:
            lp0 += cal_full_lp.GetBetaGroup(dataC, dataU, graph.GetMarkers(i), 0)
            lp1 += cal_full_lp.GetBetaGroup(dataC, dataU, graph.GetMarkers(i), 1)
        for j in range(i, graph.GetNumCliques()):
            if not graph.HasInteraction(i, j):
                continue
            c_i = graph.GetMarkers(i)
            c_j = graph.GetMarkers(j)
            lp0 += cal_gamma.CalGamma(dataC, dataU, graph, priors.pA, c_i, c_j, 0)
            lp1 += cal_gamma.CalGamma(dataC, dataU, graph, priors.pA, c_i, c_j, 1)
    return lp0, lp1



#
# Graph structure (2 cliques, and 1 components):
#
#         .------.       .---.
#        /   n0   \____ / n1  \
#        \ {0, 1} /     \ {2} /
#         '------'       '---'
#
# Connected components (modules):
#     { n0, n1 }
#
def InitializeGraph():
    g = disease_graph.Graph()
    g.AddClique([0, 1])
    g.AddClique([2])
    g.SetInteraction(0, 1)
    return g


class CalFullLPTestCase(unittest.TestCase):
    def test_CalGraphLP(self):
        g = InitializeGraph()
        num_loci = len(data_set_4.dataC[0])
        mmember = [ False for l in range(num_loci) ]
        for c in range(g.GetNumCliques()):
            for m in g.GetMarkers(c):
                mmember[m] = True
        dataU = data_set_4.dataU
        dataC = data_set_4.dataC
        genos = dataC + dataU
        lbls = [ 1 if n < len(dataC) else 0 for n in range(len(dataC) + len(dataU)) ]
        mAlpha, mBeta = calc_alpha_beta.InitAlphaBeta(genos, lbls, priors.ALPHA)
        lp0, lp1 = CalGraphLP(dataC, dataU, mmember, g, mAlpha, mBeta)
        print('test_CalGraphLP   ->   lp0:%f    lp1:%f' % (lp0, lp1))
        self.assertTrue(abs(10.9663113126 - lp0) < 1e-5)
        #self.assertTrue(abs(55.9288950713 - lp1) < 1e-5)
        self.assertTrue(abs(53.654951 - lp1) < 1e-5)


if __name__ == '__main__':
    unittest.main()

