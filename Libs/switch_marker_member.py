import random
import unittest

import numpy as np

import cal_gamma
import calc_alpha_beta
import common
import data_set_5
import disease_graph
import mix_lp
import priors
import update_marker_member



def GetSwitchDLP(dataC, dataU, graph, marker, marker_sw, mAlpha, mBeta):
    dlp0 = mAlpha[marker]
    dlp1 = mBeta[marker]
    dlp0a = mAlpha[marker_sw]
    dlp1a = mBeta[marker_sw]

    odx, ody = graph.FindMarker(marker)
    g1 = graph.GetMarkersCopy(odx)
    del g1[ody]
    if len(g1) > 0:
        dlp1 += cal_gamma.CalGamma(dataC, dataU, 1, priors.pA, g1, [ marker ], 1)
        dlp1a += cal_gamma.CalGamma(dataC, dataU, 1, priors.pA, g1, [ marker_sw], 1)

    for k in range(graph.GetNumCliques()):
        if not graph.HasInteraction(odx, k):
            continue

        g2 = graph.GetMarkers(k)
        if len(g1) > 0:
            dlp1 += cal_gamma.CalGamma(dataC, dataU, priors.pD, priors.pA, g1 + [ marker ], g2, 1)
            dlp1a += cal_gamma.CalGamma(dataC, dataU, priors.pD, priors.pA, g1 + [ marker_sw ], g2, 1)
            g_odx = graph.GetMarkers(odx)
            g_k = graph.GetMarkers(k)
            edge_gamma = cal_gamma.CalGamma(dataC, dataU, priors.pD, priors.pA, g_odx, g_k, 1)
            dlp1 -= edge_gamma
            dlp1a -= edge_gamma
        else:
            dlp1 += cal_gamma.CalGamma(dataC, dataU, priors.pD, priors.pA, g2, [ marker ], 1)
            dlp1a += cal_gamma.CalGamma(dataC, dataU, priors.pD, priors.pA, g2, [ marker_sw ], 1)
    return dlp0, dlp1, dlp0a, dlp1a


def GetSwitchDGammas(dataC, dataU, graph, marker, marker_sw, mAlpha, mBeta):
    modules = graph.GetComponents()
    num_nodes = np.sum([len(m) for m in modules])
    odx, ody = graph.FindMarker(marker)

    dgamma0 = []
    dgamma1 = []
    dgamma0a = []
    dgamma1a = []

    for component in modules:
        for clique_idx in component:
            g1 = graph.GetCliqueCopy(clique_idx)                                # Exclude marker if it is in clique_idx.
            if clique_idx == odx:
                del g1[ody]
            if len(g1) == 0:
                continue

            c_1 = [ marker ]
            c_2 = [ marker_sw ]
            dlp = cal_gamma.CalGamma(dataC, dataU, 1, 1, g1, c_1, 0)
            dlpa = cal_gamma.CalGamma(dataC, dataU, 1, 1, g1, c_2, 0)
            dgamma0.append(dlp)
            dgamma0a.append(dlpa)

            dlp += mAlpha[marker]
            dlpa += mAlpha[marker_sw]
            for k in range(graph.GetNumCliques()):
                if not graph.HasInteraction(clique_idx, k):
                    continue

                g2 = graph.GetMarkersCopy(k)
                if odx == k:
                    del g2[ody]
                if len(g2) == 0:
                    continue

                dlp += cal_gamma.CalGamma(dataC, dataU, 1, priors.pA, g1 + c_1, g2, 0)
                dlpa += cal_gamma.CalGamma(dataC, dataU, 1, priors.pA, g1 + c_2, g2, 0)
                ff = cal_gamma.CalGamma(dataC, dataU, 1, priors.pA, g1, g2, 0)
                dlp -= ff
                dlpa -= ff
            dgamma1.append(dlp)
            dgamma1a.append(dlpa)
    return dgamma0, dgamma1, dgamma0a, dgamma1a


def GetPitmanYorProbs(g, marker_id):
    odx, ody = g.FindMarker(marker_id)
    jP = []
    total_markers = 0
    num_cliques = 0
    for i in range(g.GetNumCliques()):
        num_markers = g.GetNumMarkers(i) + (-1 if i == odx else 0)
        if num_markers > 0:
            jP.append(num_markers - priors.jB)
            num_cliques += 1
            total_markers += num_markers
    jP = np.array(jP) / (total_markers + priors.jA)
    sumjP = np.sum(jP)
    pInt = num_cliques / (num_cliques + priors.pC)
    return pInt, sumjP, jP


def SwitchMarkerMember(dataC, dataU, positions, graph, marker, marker_status,
        mAlpha, mBeta, expon=random.expovariate, unif=random.uniform):
    if marker_status[marker] == False:
        return 0

    num_loci = len(marker_status)
    mu = 20000.0
    i = max(0, marker - 5)
    while i < marker and positions[i][0] != positions[marker][0]:
        i += 1
    j = min(marker + 5, num_loci - 1)
    while j > marker and positions[i][0] != positions[marker][0]:
        j += 1
    if (positions[j][1] - positions[i][1]) > (int)(mu * 2.0):
        mu = (positions[j][1] - positions[i][1]) / 2.0
    if mu > 20000.0:
        mu = 20000.0

    tries = 0
    m_chr = positions[marker][0]
    m_pos = positions[marker][1]
    while tries < 20:
        l = expon(mu)
        u = unif(0, 1)
        if u < 0.5:
            i = marker
            while i >= 0 and positions[i][0] == m_chr and positions[i][1] + l > m_pos:
                i -= 1
            if i >= 0 and marker_status[i] == False:
                rp = i
                break
        else:
            i = marker
            while i < num_loci and positions[i][0] == m_chr and positions[i][1] - l < m_pos:
                i += 1
            if i < num_loci and marker_status[i] == False:
                rp = i
                break
        tries += 1
    if tries >= 20:
        return 0

    dlp0, dlp1, dlp0a, dlp1a = GetSwitchDLP(dataC, dataU, graph, marker, rp, mAlpha, mBeta)
    dgamma0, dgamma1, dgamma0a, dgamma1a = GetSwitchDGammas(dataC, dataU, graph, marker, rp, mAlpha, mBeta)

    cdlp0 = cdlp0a = 0
    components = graph.GetComponents()
    for i in range(len(components)):
        if len(components[i]) == 0:
            continue
        p0, tmp_max0 = update_marker_member.GetSumProbMaxDgamma(dgamma0, components, i)
        p0 = priors.MINUS_INFINITE if p0 <= 0 else (np.log(p0) + tmp_max0)
        cdlp0 += mix_lp.MyGetMixLP(p0, 1.0 - priors.pA)

        p0a, tmp_max0a = update_marker_member.GetSumProbMaxDgamma(dgamma0a, components, i)
        p0a = priors.MINUS_INFINITE if p0a <= 0 else (np.log(p0a) + tmp_max0a)
        cdlp0a += mix_lp.MyGetMixLP(p0a, 1.0 - priors.pA)
    tmp = len(components) * np.log(1.0 - priors.pA)
    lg_1_tmp = np.log(1.0 - np.exp(tmp))
    cdlp0 = (np.log(1.0 - np.exp(tmp - cdlp0)) - lg_1_tmp + cdlp0) if cdlp0 > tmp else priors.MINUS_INFINITE
    cdlp0a = (np.log(1.0 - np.exp(tmp - cdlp0a)) - lg_1_tmp + cdlp0a) if cdlp0a > tmp else priors.MINUS_INFINITE

    pInt, sumjP, jP = GetPitmanYorProbs(graph, marker)

    cdlp0 = mix_lp.MyGetMixLP(cdlp0, pInt)
    dlp0 += cdlp0

    cdlp0a = mix_lp.MyGetMixLP(cdlp0a, pInt)
    dlp0a += cdlp0a

    max_lp0 = max(dgamma1) if len(dgamma1) else priors.MINUS_INFINITE
    p02 = np.sum(np.exp(dgamma1 - max_lp0) * jP) if len(dgamma1) else 0
    p02 = priors.MINUS_INFINITE if p02 <= 0 else np.log(p02) + max_lp0 - np.log(sumjP)
    dlp0 += mix_lp.MyGetMixLP(p02 - dlp0, sumjP)

    max_lp0a = max(dgamma1a) if len(dgamma1a) else priors.MINUS_INFINITE
    p02a = np.sum(np.exp(dgamma1a - max_lp0a) * jP) if len(dgamma1a) else 0
    p02a = priors.MINUS_INFINITE if p02a <= 0 else np.log(p02a) + max_lp0a - np.log(sumjP)
    dlp0a += mix_lp.MyGetMixLP(p02a - dlp0a, sumjP)

    dlp = 0
    un = unif(0, 1)
    log_un = np.log(un)
    tmp_val = dlp1a - dlp0a - dlp1 + dlp0
    if log_un < tmp_val:
        odx, ody = graph.FindMarker(marker)
        graph.cliques[odx][ody] = rp
        marker_status[marker] = False
        marker_status[rp] = True
        dlp = tmp_val
    return dlp



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


my_unif_idx = 0
my_unif_sequence = [ 10 * 20000, 0.9, 0.1 ]
def MyUnif(a, b):
    global my_unif_idx, my_unif_sequence
    s = my_unif_sequence[my_unif_idx]
    my_unif_idx = (my_unif_idx + 1) % len(my_unif_sequence)
    u = s / (b - a)
    return u


def MyExpoVar(par_lambda):
    return MyUnif(0, par_lambda)


class SwitchMarkerMemberTestCase(unittest.TestCase):
    def test_GetSwitchDLP(self):
        g = InitializeGraph()
        dataU = data_set_5.dataU
        dataC = data_set_5.dataC
        genos = dataC + dataU
        lbls = [ 1 if n < len(dataC) else 0 for n in range(len(dataC) + len(dataU)) ]
        mAlpha, mBeta = calc_alpha_beta.InitAlphaBeta(genos, lbls, priors.ALPHA)
        dlp0, dlp1, dlp0a, dlp1a = GetSwitchDLP(dataC, dataU, g, 2, 3, mAlpha, mBeta)
        print('dlp0:%f     dlp1:%f     dlp0a:%f     dlp1a:%f' % (dlp0, dlp1, dlp0a, dlp1a))
        self.assertTrue(abs(-80.30848954746205 - dlp0) < 1e+7)
        self.assertTrue(abs(-34.218395918692714 - dlp1) < 1e+7)
        self.assertTrue(abs(-60.188167648307115 - dlp0a) < 1e+7)
        self.assertTrue(abs(-17.58021734448796 - dlp1a) < 1e+7)


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


    def test_GetSwitchDGammas(self):
        EXP_DGAMMA_0  = [34.584361256121014]
        EXP_DGAMMA_1  = [-45.72412829134103]
        EXP_DGAMMA_0A = [32.82980899519106]
        EXP_DGAMMA_1A = [-27.358358653116056]
        EXP_DGAMMAS = [ EXP_DGAMMA_0, EXP_DGAMMA_1, EXP_DGAMMA_0A, EXP_DGAMMA_1A ]

        g = InitializeGraph()
        dataU = data_set_5.dataU
        dataC = data_set_5.dataC
        genos = dataC + dataU
        lbls = [ 1 if n < len(dataC) else 0 for n in range(len(dataC) + len(dataU)) ]
        mAlpha, mBeta = calc_alpha_beta.InitAlphaBeta(genos, lbls, priors.ALPHA)
        dg0, dg1, dg0a, dg1a = GetSwitchDGammas(dataC, dataU, g, 2, 3, mAlpha, mBeta)
        self.CheckDgammaEq([dg0, dg1, dg0a, dg1a], EXP_DGAMMAS)


    def test_SwitchMarkerMember(self):
        num_loci = len(data_set_5.dataC[0])
        dataU = data_set_5.dataU
        dataC = data_set_5.dataC
        counts = [ [ common.CountCombinations(dataU, [i]),
                    common.CountCombinations(dataC, [i]) ] for i in range(num_loci) ]
        mAlpha, mBeta = calc_alpha_beta.CalcAlphaBeta(counts, priors.ALPHA)

        g = InitializeGraph()                                                   # Initialize disease graph.
        mmember = [ False for l in range(num_loci) ]
        for c in range(g.GetNumCliques()):
            for m in g.GetMarkers(c):
                mmember[m] = True
        positions = [ (l, 1 if l < 4 else 2) for l in range(num_loci) ]
        print('After initialize . . .')
        print(g.ToString())
        print(mmember)
        print(positions)
        print('')

        dlp = SwitchMarkerMember(dataC, dataU, positions, g, 2, mmember, mAlpha, mBeta, MyExpoVar, MyUnif)
        self.assertTrue(abs(-1.7275910640201886 - dlp) < 1e-5)
        self.assertTrue(mmember[3])
        self.assertFalse(mmember[2])
        print('After update marker member . . .')
        print(g.ToString())
        print('dlp:{}    new_mmember:{}'.format(dlp, mmember))


if __name__ == '__main__':
    unittest.main()

