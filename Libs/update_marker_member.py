import random
import unittest

import numpy as np
import scipy.special as sp

import cal_gamma
import calc_alpha_beta
import common
import data_set_4
import disease_graph
import get_dlp
import mix_lp

import priors



# Finds max dgamma of i-th component.
# Returns sum of exp(dgamma[k : k + comp_len] - mx)
def GetSumProbMaxDgamma(dgamma, components, i):
    k = np.sum([ len(components[j]) for j in range(0, i - 1) ], dtype=int)
    dg = dgamma[k : k + len(components[i])]
    mx = max(dg) if len(dg) else priors.MINUS_INFINITE
    prb = np.sum(np.exp(dg - mx)) if len(dg) else priors.MINUS_INFINITE
    return prb, mx


def GetExpContrib(x, logrp, idx):
    alpha = x[idx]
    beta = 0
    tgamma = []
    tlp = []
    clp = []
    for i in range(len(logrp[idx])):
        if logrp[idx][i] < 0:
            tlp.append(logrp[idx][i])
            clp.append(np.log(-logrp[idx][i]) if logrp[idx][i] > -1e-10 else np.log(1.0 - np.exp(logrp[idx][i])))
            logrp[idx][i] = logrp[i][idx] = 0
            a, b = GetExpContrib(x, logrp, i)
            tgamma.append(a - b)
            beta += b
    if len(tgamma) > 0:
        tmax = max([alpha] + tgamma)
        sumclp = np.sum(clp)
        sm = np.exp(alpha - tmax) + np.sum(np.exp(tgamma - tmax))
        cuma = [ beta + sumclp + np.log(sm) + tmax ]
        cumb = [ beta + sumclp ]
        for i in range(1, len(tgamma) + 1):
            lst = range(0, i)
            while True:
                tmp = np.sum([beta + sumclp] + [tlp[lst[j]] - clp[lst[j]] + tgamma[lst[j]] for j in range(len(lst))])
                cumb.append(tmp)

                tmax = max([alpha] + [tgamma[j] for j in range(len(tgamma)) if j not in lst])
                sm = np.exp(tmax - alpha) + np.sum([np.exp(tgamma[j] - tmax) for j in range(len(tgamma)) if j not in lst])

                tmp += np.log(sm) + tmax
                cuma.append(tmp)

                # Next combination
                j = j - 1
                lst[j] += 1
                while lst[j] >= len(tgamma) - (i - 1 - j):
                    j -= 1
                    if j < 0:
                        break
                    lst[j] += 1
                if j >= 0:
                    for k in range(j + 1, i):
                        lst[k] = lst[k - 1] + 1
                if j < 0:
                    break

        tmaxa = max(cuma)
        alpha = np.sum(np.exp(cuma - tmaxa))
        alpha = np.log(alpha) + tmaxa

        tmaxb = max(cumb)
        sm = np.sum(np.exp(cumb - tmaxb))
        beta = np.log(sm) + tmaxb
    return alpha, beta


def CalExpContrib(dataC, dataU, graph, components, dgamma, calc_type):
    cdlp = 0.0
    lp = []
    k = 0
    for i in range(len(components)):
        sz = len(components[i])
        logrp = np.zeros((sz, sz), dtype=float)                                 # Log probability of removing an edge, if 0 mean no connection
        for j in range(sz - 1):                                                 # The sz - 1 might be BAGGUS if components[i] is not sorted ???
            nid = components[i][j]
            for l in range(graph.GetNumCliques()):
                if not graph.HasInteraction(nid, l) or l < nid:
                    continue

                m = components[i].index(l)                                      # TODO: Solve ValueError when searching item does not exist in list!
                g1 = graph.GetMarkers(nid)
                g2 = graph.GetMarkers(components[i][m])
                p = cal_gamma.CalGamma(dataC, dataU, 1.0, priors.pA, g1, g2, calc_type)
                if calc_type == 0:
                    p = np.log(1.0 - priors.pA) - mix_lp.MyGetMixLP(p, priors.pA)
                else:
                    p = np.log(1.0 - priors.pD) - mix_lp.MyGetMixLP(p, priors.pD)
                logrp[j][m] = logrp[m][j] = p
        x = dgamma[k : k + sz]
        a, b = GetExpContrib(x, logrp, 0)
        a = mix_lp.MyGetMixLP(a, priors.pA if calc_type == 0 else priors.pD)
        lp.append(a)
        cdlp += a
        k += sz
    return cdlp, lp


def MakeProbAndSelectOption(marker, dlp0, dlp1, dgamma12, jP, graph):
    max_lp12 = max(dgamma12) if len(dgamma12) else priors.MINUS_INFINITE
    max_lp = max([ max_lp12, dlp0, dlp1 ])
    prob = np.zeros((len(dgamma12) + 2), dtype=float)
    prob[0] = np.exp(dlp0 - max_lp)
    prob[1] = prob[0] + np.exp(dlp1 - max_lp)
    module_idx, marker_idx = graph.FindMarker(marker)
    for i in range(len(dgamma12)):
        if graph.GetNumMarkers(i) - (1 if i == module_idx else 0) < priors.mBound / 2:
            prob[i + 2] = np.exp(dgamma12[i] - max_lp + np.log(jP[i]))
        prob[i + 2] += prob[i + 1]

    un = random.uniform(0, prob[-1])
    for i in range(len(prob)):
        if un < prob[i]:
            return i
    return len(prob) - 1


def GetCliqueIdx(graph, marker, idx):
    nto = -1
    odx, ody = graph.FindMarker(marker)
    if idx == 1:
        if odx == None or graph.GetNumMarkers(odx) > 1:
            nto = graph.GetNumCliques()
        else:
            nto = odx
    else:
        nto = idx - 2
    return nto, odx, ody


def GetComponentIdx(components, clique_idx):
    idx = 0
    for i in range(len(components)):
        if clique_idx in components[i]:
            return idx + components[i].index(clique_idx)
        idx += len(components[i])
    return None


def UpdateMarkerMember(dataC, dataU, graph, marker, marker_status, mAlpha, mBeta):
    dgamma0, dgamma1, dgamma02, dgamma12 = get_dlp.GetDLP(dataC, dataU, graph, marker, mAlpha, mBeta)

    jP = np.array([ graph.GetNumMarkers(i) - priors.jB for i in range(graph.GetNumCliques()) ])     # Pitman-Yor probabilities
    jP /= graph.GetTotalNumMarkers() + priors.jA
    sumjP = np.sum(jP)
    pInt = graph.GetNumCliques() / (graph.GetNumCliques() + priors.pC)

    components = graph.GetComponents()
    ip0 = np.zeros((len(components)), dtype=float)
    ip1 = np.zeros((len(components)), dtype=float)
    #cdlp0 = 0.0; cdlp1 = 0.0
    for i in range(len(components)):
        p0, tmp_max0 = GetSumProbMaxDgamma(dgamma0, components, i)
        ip0[i] = -np.Inf if p0 < 0 else (np.log(p0 / len(components[i])) + tmp_max0)
        #cdlp0 += mix_lp.MyGetMixLP(ip0[i], priors.pD)

        p1, tmp_max1 = GetSumProbMaxDgamma(dgamma1, components, i)
        ip1[i] = -np.Inf if p1 < 0 else (np.log(p1 / len(components[i])) + tmp_max1)
        #cdlp1 += mix_lp.MyGetMixLP(ip1[i], priors.pA)

    cdlp0, lp0 = CalExpContrib(dataC, dataU, graph, components, dgamma0, 0)
    cdlp1, lp1 = CalExpContrib(dataC, dataU, graph, components, dgamma1, 1)

    tmp0 = len(components) * np.log(1.0 - priors.pA)
    cdlp0 = np.log(1.0 - np.exp(tmp0 - cdlp0)) - np.log(1.0 - np.exp(tmp0)) + cdlp1 if cdlp0 > tmp0 else priors.MINUS_INFINITE
    cdlp0 = mix_lp.MyGetMixLP(cdlp0, pInt)
    dlp0 = mAlpha[marker] + cdlp0
    max_lp02 = max(dgamma02) if len(dgamma02) else priors.MINUS_INFINITE
    p02 = np.sum(np.exp(dgamma02 - max_lp02) * jP) if len(dgamma02) else 0
    p02 = priors.MINUS_INFINITE if p02 <= 0 else np.log(p02) + max_lp02 - np.log(sumjP)
    dlp0 += mix_lp.MyGetMixLP(p02 - dlp0, sumjP)
    if marker_status == True:
        dlp0 = priors.MINUS_INFINITE
    dlp0 -= np.log(priors.pI / (1.0 - priors.pI))

    tmp1 = len(components) * np.log(1.0 - priors.pD)
    cdlp1 = np.log(1.0 - np.exp(tmp1 - cdlp1)) - np.log(1.0 - np.exp(tmp1)) + cdlp1 if cdlp1 > tmp1 else priors.MINUS_INFINITE
    cdlp1 = mix_lp.MyGetMixLP(cdlp1, pInt)
    dlp1 = mBeta[marker] + cdlp1 + np.log(1.0 - sumjP)

    idx = MakeProbAndSelectOption(marker, dlp0, dlp1, dgamma12, jP, graph)
    dlp = 0
    new_mstatus = False
    if idx != 0 or marker_status:
        nto, odx, ody = GetCliqueIdx(graph, marker, idx)

        if odx != nto:                                                          # Node structure change
            if odx != None:
                graph.RemoveMarkerFromClique(odx, ody)                          # Remove marker from graph.
                if graph.GetNumMarkers(odx) > 1:
                    l = GetComponentIdx(components, odx)
                    dlp -= dgamma12[l] + jP[l]
                else:
                    dlp -= dlp1
                dlp -= np.log(priors.pI / (1.0 - priors.pI))

            if nto >= 0:                                                        # Add marker to graph.
                graph.AddMarker(nto, marker)
                if idx == 1:
                    dlp += dlp1
                else:
                    dlp += dgamma12[idx - 2] + np.log(jP[idx - 2])
                dlp += np.log(priors.pI / (1.0 - priors.pI))
                new_mstatus = True
            else:
                new_mstatus = False

        if idx == 1:                                                            # Update connection if i forms a node by itself
            graph.RemoveInteractions(nto)
            u = random.uniform(0, 1)
            log_u = np.log(u)
            log_1_pInt = np.log(1.0 - pInt)
            if pInt == 1.0 or log_u > log_1_pInt - cdlp1:
                k = 0
                for j in range(len(components)):
                    mmp = max(ip1[j], 0.0)
                    p0 = (1.0 - priors.pD) * np.exp(-mmp)

                    end_prb = np.exp(np.log(p0 + priors.pD * np.exp(ip1[j] - mmp)))
                    un = random.uniform(0, end_prb)
                    if un > np.exp(np.log(p0)):
                        #mmp0 = max(dgamma0[k : k + len(components[j])])
                        mmp1 = max(dgamma1[k : k + len(components[j])])
                        cum_sum = np.array(np.exp(dgamma1[k : k + len(components[j])] - mmp1))
                        for l in range(1, len(cum_sum)):
                            cum_sum[l] += cum_sum[l - 1]
                        un = random.uniform(0, cum_sum[-1])
                        for l in range(len(cum_sum)):
                            if un <= cum_sum[l]:
                                break
                        graph.SetInteraction(components[j][l], nto)
                    k += len(components[j])
    return dlp, new_mstatus



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


def CalcLogOfLogRP(logrp, init_mode=0):
    zero_val = -23 if init_mode == 1 else 0
    for i in range(len(logrp)):
        for j in range(len(logrp[i])):
            if i < j:
                logrp[i][j] = zero_val if logrp[i][j] == 0 else np.log(logrp[i][j])
            else:
                logrp[i][j] = logrp[j][j]
    return np.array(logrp)


class UpdateMarkerMemberTestCase(unittest.TestCase):
    def test_UpdateMarkerMember(self):
        num_loci = len(data_set_4.dataC[0])
        dataU = data_set_4.dataU
        dataC = data_set_4.dataC
        counts = [ [ common.CountCombinations(dataU, [i]),
                    common.CountCombinations(dataC, [i]) ] for i in range(num_loci) ]
        mAlpha, mBeta = calc_alpha_beta.CalcAlphaBeta(counts, priors.ALPHA)

        g = InitializeGraph()                                                   # Initialize disease graph.
        print('After initialize . . .')
        print(g.ToString())
        print('')

        dlp, new_mmember = UpdateMarkerMember(dataC, dataU, g, 3, False, mAlpha, mBeta)
        print('test_UpdateMarkerMember -> dlp:%f' % dlp)
        #self.assertFalse(abs(-24.20043 - dlp) > 1e-5)
        self.assertTrue(new_mmember)
        print('After update marker member . . .')
        print(g.ToString())
        print('dlp:{}    new_mmember:{}'.format(dlp, new_mmember))


    def test_GetContribFunc(self):
        #
        #
        #      .--. 0.99 .--.
        #     ( n0 )----( n1 )
        #      '--'      '--'
        #                  | 0.5
        #                .--.
        #               ( n2 )
        #                '--'
        #
        #
        logrp = CalcLogOfLogRP([
            [ 0.00, 0.99, 0.00 ],
            [ 0.00, 0.00, 0.50 ],
            [ 0.00, 0.00, 0.00 ],
        ])
        x = np.log(np.array([0.99, 0.1, 0.2]))                                  # Probability of supper nodes, or dgamma of component.
        alpha, beta = GetExpContrib(x, logrp, 0)
        self.assertFalse(abs(-1.80572 - alpha) > 1e-5 and abs(-1.80545 - beta) > 1e-5)
        print('alpha : {}'.format(alpha))
        print('beta  : {}'.format(beta))
        if abs(-1.80572 - alpha) > 1e-5 or abs(-1.80545 - beta) > 1e-5:
            print('FAILED', abs(-1.80572 - alpha), abs(-1.80545 - beta))
        else:
            print('OK!')


if __name__ == '__main__':
    unittest.main()

