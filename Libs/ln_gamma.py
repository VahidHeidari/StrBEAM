import unittest

import numpy as np
import scipy.special as sp



Alphan = 1.5
alleleCn = 3
N = 6
blockMax = 5



def BEAMInitGamma(N, blockMax, Alphan, alleleCn):
    lgs = []
    for i in range(blockMax):
        lgs.append([])
        delta = Alphan / (3.0 ** i)
        p = sp.loggamma(delta)
        neg_p = -p
        for j in range(N):
            lg = p + neg_p
            p += np.log(j + delta)
            lgs[i].append(lg)
    return lgs



def MyInitGamma(N, max_block, alpha_0, num_alleles):
    lgs = []
    for k in range(max_block):                                                  # k \in { 1, ..., 3^K }
        lgs.append([])
        p = alpha_0 / (3 ** k)
        lgg_p = sp.loggamma(p)
        for n_i in range(N):                                                    # n_i \in { 0, ..., N }
            lg = sp.loggamma(n_i + p) - lgg_p
            lgs[k].append(lg)
    return lgs



class InitLnGammaTestCase(unittest.TestCase):
    def PrintAndCheckLnGamma(self, msg, N, max_block, alpha_0, num_alleles, lg):
        exp_lg = [
            [ 0.0000000,  0.4054651,  1.3217558,  2.5745188,  4.0785962,  5.7833443 ],
            [ 0.0000000, -0.6931472, -0.2876821,  0.6286087,  1.8813716,  3.3854490 ],
            [ 0.0000000, -1.7917595, -1.6376088, -0.8644189,  0.2882606,  1.7153770 ],
            [ 0.0000000, -2.8903718, -2.8363045, -2.1157584, -0.9987970,  0.4012907 ],
            [ 0.0000000, -3.9889840, -3.9706349, -3.2682711, -2.1635049, -0.7725916 ],
        ]

        out_str = msg + '\n'
        out_str += '    N           : {}\n'.format(N)
        out_str += '    max_block   : {}\n'.format(max_block)
        out_str += '    alpha_0     : {}\n'.format(alpha_0)
        out_str += '    num_alleles : {}\n'.format(num_alleles)
        for i in range(len(lg)):
            out_str += '[{:3d}]   '.format(i)
            for j in range(len(lg[i])):
                out_str += ' {:>7.3f}'.format(lg[i][j])
                self.assertFalse(abs(lg[i][j] - exp_lg[i][j]) > 1e-5)
            out_str += '\n'
        print(out_str)


    def test_InitLnGamma(self):
        beam_lg = BEAMInitGamma(N, blockMax, Alphan, alleleCn)
        self.PrintAndCheckLnGamma('BEAM Implementation For Initialize Gamma Function:',
                N, blockMax, Alphan, alleleCn, beam_lg)

        my_lg = MyInitGamma(N, blockMax, Alphan, alleleCn)
        self.PrintAndCheckLnGamma('My Implementation For Initialize Gamma Function:',
                N, blockMax, Alphan, alleleCn, beam_lg)


if __name__ == '__main__':
    unittest.main()

