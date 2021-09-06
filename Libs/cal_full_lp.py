import unittest

import numpy as np
import scipy.special as sp

import BEAM
import cal_gamma
import calc_alpha_beta
import data_set_4
import disease_graph
import priors



def GetLogGamma(k, n, alpha_0):
    p = alpha_0 / (3.0 ** k)
    lgg_p = sp.loggamma(p)
    lg = sp.loggamma(n + p) - lgg_p
    return lg


def GetBetaGroup(dataC, dataU, markers, calc_type=1):
    if len(markers) == 0:
        return 0

    genos = dataC + dataU
    lbls = [ 1 if n < len(dataC) else 0 for n in range(len(dataC) + len(dataU)) ]
    cnts = BEAM.CountCombAlleles(genos, lbls, markers)
    k = len(markers)
    rt = 0
    for i in range(len(cnts[0])):
        if calc_type == 1:
            l1 = GetLogGamma(k, cnts[0][i], priors.ALPHA)
            l2 = GetLogGamma(k, cnts[1][i], priors.ALPHA)
            rt += l1 + l2
        else:
            c = cnts[0][i] + cnts[1][i]
            rt += GetLogGamma(k, c, priors.ALPHA)
    if calc_type == 1:
        l1 = GetLogGamma(0, len(dataC), priors.ALPHA)
        l2 = GetLogGamma(0, len(dataU), priors.ALPHA)
        rt -= l1 + l2
    else:
        rt -= GetLogGamma(0, len(genos), priors.ALPHA)
    for i in range(len(markers)):
        cnts = BEAM.CountSingleLocusAlleles(genos, lbls, markers[i])
        drt = 0
        if calc_type == 1:
            l1 = GetLogGamma(1, cnts[0][0], priors.ALPHA)
            l2 = GetLogGamma(1, cnts[1][0], priors.ALPHA)
            drt -= l1 + l2
            l1 = GetLogGamma(0, len(dataC), priors.ALPHA)
            l2 = GetLogGamma(0, len(dataU), priors.ALPHA)
            drt += l1 + l2
        else:
            c = cnts[0][0] + cnts[1][0]
            drt -= GetLogGamma(1, c, priors.ALPHA)
            drt += GetLogGamma(0, len(genos), priors.ALPHA)
        rt += drt
    return rt


def CalFullLP(dataC, dataU, mmember, graph, mAlpha, mBeta):
    num_loci = len(mAlpha)
    lp = 0
    lg_pI_1 = np.log(priors.pI)
    lg_pI_0 = np.log(1.0 - priors.pI)
    for i in range(num_loci):
        lp += (mAlpha[i] + lg_pI_0) if mmember[i] == False else (mBeta[i] + lg_pI_1)

    for i in range(graph.GetNumCliques()):
        if graph.GetNumMarkers(i) > 1:
            lp += GetBetaGroup(dataC, dataU, graph.GetMarkers(i))
        c_i = graph.GetMarkers(i)
        for j in range(i, graph.GetNumCliques()):
            if not graph.HasInteraction(i, j):
                continue
            c_j = graph.GetMarkers(j)
            lp += cal_gamma.CalGamma(dataC, dataU, 1, priors.pA, c_i, c_j, 1)
    return lp



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


# TODO: Add a test for GetBetaGroup.
class CalFullLPTestCase(unittest.TestCase):
    def test_CalFullLP(self):
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
        lp = CalFullLP(dataC, dataU, mmember, g, mAlpha, mBeta)
        print('test_CalFullLP -> lp:%f' % lp)
        self.assertTrue(abs(-13.5463325 - lp) < 1e-5)


if __name__ == '__main__':
    unittest.main()

