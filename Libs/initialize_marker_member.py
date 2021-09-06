import random

import numpy as np

import cal_gamma
import disease_graph
import priors



class RiskRec:
    def __init__(self, idx, score):
        self.index = idx
        self.score = score


    def __str__(self):
        return '(index:{}, score:{})'.format(self.index, self.score)


def CompRiskRec(a, b):
    if a.score < b.score:
        return 1
    if a.score > b.score:
        return -1
    if a.index < b.index:
        return 1
    if a.index > b.index:
        return -1
    return 0


def Init(genos, labels, positions, mAlpha, mBeta):
    num_indivs = len(genos)
    num_loci = len(mAlpha)
    m_status = [ False for l in range(num_loci) ]
    graph = disease_graph.Graph()

    risk_list = [ RiskRec(i, mBeta[i] - mAlpha[i]) for i in range(num_loci) if mBeta[i] - mAlpha[i] > 1.0 ]
    risk_list = sorted(risk_list, cmp=CompRiskRec)
    i = len(risk_list) - 1
    while i >= 0:
        chrom = positions[risk_list[i].index][0]
        pos = positions[risk_list[i].index][1]
        for j in range(i + 1, len(risk_list)):
            rg = abs(pos - positions[risk_list[j].index][1])
            if chrom == positions[risk_list[j].index][0] and rg < 5000 * risk_list[j].score:
                del risk_list[i]
                break
        i -= 1

    for r in risk_list:
        m_status[r.index] = True

    # Initialize marker status and cliques.
    d_n = 0
    for i in range(num_loci):
        un = random.uniform(0, 1)
        if un < min(priors.pI, 20.0 / num_loci):
            m_status[i] = True

            un_new = random.uniform(0, 1)                                       # Pitman-Yor process.
            new_prob = (priors.jA + graph.GetNumCliques() * priors.jB) / (d_n + priors.jA)
            if un_new <= new_prob:
                graph.AddClique([i])                                            # Create new clique.
            else:
                k = random.choice(range(graph.GetNumCliques()))                 # Add to existing clique.
                if graph.GetNumMarkers(k) <= priors.mBound / 2:
                    graph.AddMarker(k, i)
                else:
                    graph.AddClique([i])
            d_n += 1

    # initialize interactions.
    modules = [ [i] for i in range(graph.GetNumCliques()) ]
    dataU = [ genos[n] for n in range(num_indivs) if labels[n] == 0 ]
    dataC = [ genos[n] for n in range(num_indivs) if labels[n] == 1 ]
    for r in range(graph.GetNumCliques()):
        i = random.choice(range(len(modules)))
        j = random.choice(range(len(modules[i])))
        k = random.choice(range(len(modules)))
        if k == i:
            continue

        tlp = np.zeros(len(modules[k]))
        for l in range(len(modules[k])):
            c_ij = graph.GetMarkers(modules[i][j])
            c_kl = graph.GetMarkers(modules[k][l])
            tlp[l] = cal_gamma.CalGamma(dataC, dataU, 1, priors.pA, c_ij, c_kl, 1)
        max_lp = max(tlp)
        tlp = priors.pD / (1 - priors.pD) * np.exp(tlp - max_lp) / len(tlp)
        lp = np.sum(tlp)
        if lp <= 0:
            continue

        lp = lp / (lp + np.exp(-max_lp))
        un_2 = random.uniform(0, 1)
        if un_2 < lp:
            tlp_cumsum = np.cumsum(tlp)
            un_3 = random.uniform(0, tlp[-1])
            for l in range(len(tlp)):
                if un_3 <= tlp[l]:
                    break
            graph.SetInteraction(modules[i][j], modules[k][l])
            modules[k] += modules[i]
            del modules[k]
    return graph, m_status



if __name__ == '__main__':
    print('This is a module!')

