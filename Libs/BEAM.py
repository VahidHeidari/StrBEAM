import numpy as np
import scipy.special as sp



def GetLogMarginalCatDir(counts, params):
    al = np.sum(params)
    sm = np.sum(counts)
    rs = np.sum(sp.loggamma(np.sum([counts, params], 0))) -                 \
            np.sum(sp.loggamma(params)) +                                   \
            sp.loggamma(al) - sp.loggamma(sm + al)
    return rs


def GetP_D0(cnts, prior, I):
    p = 0.0
    for i in I:
        sm = np.sum([cnts[i][0], cnts[i][1]], 0)
        p += GetLogMarginalCatDir(sm, prior)
    return p


def GetP_D1(cnts, prior, I):
    p = 0.0
    for i in I:
        p += GetLogMarginalCatDir(cnts[i][0], prior)
        p += GetLogMarginalCatDir(cnts[i][1], prior)
    return p


def CountSingleLocusAlleles(genos, labels, locus):
    num_indivs = len(labels)
    counts = np.zeros((2, 3), dtype=int)
    for i in range(num_indivs):
        counts[labels[i]][genos[i][locus]] += 1
    return counts


def GetComb(genos):
    cmb = 0
    for l in range(len(genos)):
        cmb *= 3
        cmb += genos[len(genos) - l - 1]
    return cmb


def CountCombAlleles(genos, labels, loci_list):
    num_indivs = len(labels)
    counts = np.zeros((2, 3 ** len(loci_list)), dtype=int)
    for i in range(num_indivs):
        snps = [ genos[i][l] for l in loci_list ]
        cmb = GetComb(snps)
        counts[labels[i]][cmb] += 1
    return counts


if __name__ == '__main__':
    print('This is a module!')

