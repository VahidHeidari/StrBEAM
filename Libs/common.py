import numpy as np
import scipy.special as sp

import priors



def CountAlleles(dataC, dataU):
    num_loci = len(dataC[0])
    cnt_u = [ [0 for l in range(priors.NUM_ALLELES) ] for j in range(num_loci) ]
    cnt_c = [ [0 for l in range(priors.NUM_ALLELES) ] for j in range(num_loci) ]
    for i in dataU:
        for j in range(len(i)):
            g = i[j]
            cnt_u[j][g] += 1

    for i in dataC:
        for j in range(len(i)):
            g = i[j]
            cnt_c[j][g] += 1

    cnts =[]
    for i in range(num_loci):
        cnts.append([ cnt_u[i], cnt_c[i] ])
    return cnts


def GetComb(genos):
    cmb = 0
    for l in range(len(genos)):
        cmb *= priors.NUM_ALLELES
        cmb += genos[l]
    return cmb


def CountCombinations(genos, loci):
    num_combs = priors.NUM_ALLELES ** len(loci)
    cnt = np.zeros((num_combs), dtype=int)                                      # Count combinations.
    for indiv in genos:
        x = [ indiv[l] for l in loci ]
        c = GetComb(x)
        cnt[c] += 1
    return cnt


def GetLogProbDirCat(cnt, k, prior=priors.ALPHA):
    p = prior / (3 ** k)
    lgg_p = sp.loggamma(p)
    lg = 0
    Nc = 0
    for n_i in cnt:
        lg += sp.loggamma(n_i + p) - lgg_p
        Nc += n_i
    lg += sp.loggamma(prior) - sp.loggamma(Nc + prior)
    return lg



if __name__ == '__main__':
    print('This is module!')

