import unittest

import numpy as np
import scipy.special as sp

import BEAM
import logger



ALPHA = 1.5

# Values calculated by original BEAM3 source code
mAlpha = [ -78.8678191874547,  -74.272699337320091, -74.272699337320091 ]
mBeta  = [ -13.825360883817154, -9.230241033682546,  -9.230241033682546 ]

# Allele counts from original BEAM3 source code
allele_counts = [
    [ [49, 1, 0], [0, 0, 50] ],
    [ [50, 0, 0], [0, 0, 50] ],
    [ [50, 0, 0], [0, 0, 50] ],
]



def VectToString(v):
    if len(v) == 0:
        return ''

    out_str = '{:>7.3f}'.format(v[0])
    for i in range(1, len(v)):
        out_str += ' {:>7.3f}'.format(v[i])
    return out_str


def CalcAlphaBeta(counts, alpha_0):
    num_loci = len(counts)
    m_alpha = np.zeros((num_loci), dtype=float)
    m_beta = np.zeros((num_loci), dtype=float)
    lgg_A = sp.loggamma(alpha_0)

    for i in range(num_loci):
        u = counts[i][0]
        c = counts[i][1]
        nu = np.sum(u)
        nc = np.sum(c)
        num_alleles = len(c)
        alpha = alpha_0 / (num_alleles ** 1)
        lgg_a = sp.loggamma(alpha)                                              # LogGamma(alpha)
        for j in range(num_alleles):
            m_alpha[i] += sp.loggamma(c[j] + u[j] + alpha) - lgg_a
            m_beta[i] += sp.loggamma(c[j] + alpha) + sp.loggamma(u[j] + alpha) - (lgg_a * 2)
        m_alpha[i] += lgg_A - sp.loggamma(nu + nc + alpha_0)
        m_beta[i] += (lgg_A * 2) - sp.loggamma(nu + alpha_0) - sp.loggamma(nc + alpha_0)
    return m_alpha, m_beta


def InitAlphaBeta(genos, labels, alpha_0):
    num_indivs = len(genos)
    if num_indivs == 0:
        return [], []

    # Check consistency.
    if num_indivs != len(labels):
        logger.Log('Warning: InitAlphaBeta()    num_indivs({}) != num_labels({})'.format(num_indivs, len(labels)))

    num_loci = len(genos[0])
    allele_counts = []
    for l in range(num_loci):
        cnt = BEAM.CountSingleLocusAlleles(genos, labels, l)
        allele_counts.append(cnt)
    mAlpha, mBeta = CalcAlphaBeta(allele_counts, alpha_0)
    return mAlpha, mBeta



class CalcAlphaBetaTestCase(unittest.TestCase):
    def test_CalcAlphaBetaInit(self):
        print('mAlpha  : ' + VectToString(mAlpha))
        print('mBeta   : ' + VectToString(mBeta))
        print('')

        m_alpha, m_beta = CalcAlphaBeta(allele_counts, ALPHA)
        print('m_alpha : ' + VectToString(m_alpha))
        print('m_beta  : ' + VectToString(m_beta))

        self.assertEqual(len(mAlpha), len(m_alpha))
        for i in range(len(mAlpha)):
            self.assertFalse(abs(mAlpha[i] - m_alpha[i]) > 1e-5)

        self.assertEqual(len(mBeta), len(m_beta))
        for i in range(len(mBeta)):
            self.assertFalse(abs(mBeta[i] - m_beta[i]) > 1e-5)

        self.assertEqual(len(mBeta), len(mAlpha))
        self.assertEqual(len(m_beta), len(m_alpha))
        self.assertEqual(len(m_beta), len(mBeta))


if __name__ == '__main__':
    unittest.main()

