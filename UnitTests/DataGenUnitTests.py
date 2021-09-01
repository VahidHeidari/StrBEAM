import inspect
import unittest

import numpy as np

import Libs.BEAM



IS_DEBUG = False



def Log(msg):
    if IS_DEBUG:
        print(str(msg))



class DataGenUnitTests(unittest.TestCase):
    def TestProbCatDir(self, count1, count2, prior):
        Log((count1, count2, prior))
        r1 = Libs.BEAM.GetLogMarginalCatDir(count1, prior)
        r2 = Libs.BEAM.GetLogMarginalCatDir(count2, prior)
        fmt = 'r1:{}    r2:{}\n' +                                              \
              '   exp(r1)={}    exp(r2)={}\n' +                                 \
              '   r1 < r2 {}\n' +                                               \
              '   r1 / r2 {}\n'
        Log(fmt.format(r1, r2, np.exp(r1), np.exp(r2), r1 < r2, np.exp(r1 - r2)))


    def TestProbIndepCatDir(self, count1, count2, prior):
        r1 = Libs.BEAM.GetLogMarginalCatDir(count1, prior)
        r2 = Libs.BEAM.GetLogMarginalCatDir(count2, prior)
        count12 = np.sum([count1, count2], 0).tolist()
        r3 = Libs.BEAM.GetLogMarginalCatDir(count12, prior)
        fmt = 'count1:{} count2:{} prior:{}\n' +                                \
              '   r1:{}    r2:{}   r1+r2:{}    r3:{}\n' +                       \
              '   exp(r1)={}    exp(r2)={}    exp(r1 + r2)={}    exp(r3)={}\n' +\
              '   Disease Locus (r3 < r1 + r2) ? {}\n'
        Log(fmt.format(count1, count2, prior,
            r1, r2, r1 + r2, r3,
            np.exp(r1), np.exp(r2), np.exp(r1 + r2), np.exp(r3),
            r3 < r1 + r2))


    def test_CatDir(self):
        self.TestProbCatDir([0, 1, 0], [0, 0, 1], [0.2, 0.6, 0.2])
        self.TestProbCatDir([10, 10, 10], [1, 28, 1], [0.5, 0.5, 0.5])
        self.TestProbCatDir([20, 60, 20], [10, 30, 10], [1, 1, 1])
        self.TestProbCatDir([1, 8, 1], [2, 16, 2], [1, 1, 1])
        self.TestProbCatDir([2, 2, 2], [1, 4, 1], [1, 1, 1])
        self.TestProbCatDir([2, 2, 2], [4, 4, 4], [1, 1, 1])

        self.TestProbIndepCatDir([2, 2, 2], [2, 2, 2], [1, 1, 1])
        self.TestProbIndepCatDir([2, 2, 2], [1, 4, 1], [1, 1, 1])
        self.TestProbIndepCatDir([5, 3, 2], [50, 20, 30], [1, 1, 1])
        self.TestProbIndepCatDir([50, 30, 20], [80, 19, 1], [1, 1, 1])


    def CheckComb(self, exp, cmb):
        res = Libs.BEAM.GetComb(cmb)
        self.assertEqual(exp, res)
        return exp == res


    def test_CombinationCount(self):
        self.CheckComb( 0, [0, 0, 0])
        self.CheckComb( 1, [1, 0, 0])
        self.CheckComb( 2, [2, 0, 0])
        self.CheckComb( 3, [0, 1, 0])
        self.CheckComb( 4, [1, 1, 0])
        self.CheckComb( 5, [2, 1, 0])
        self.CheckComb( 6, [0, 2, 0])
        self.CheckComb( 7, [1, 2, 0])
        self.CheckComb( 8, [2, 2, 0])

        self.CheckComb( 9, [0, 0, 1])
        self.CheckComb(10, [1, 0, 1])
        self.CheckComb(11, [2, 0, 1])
        self.CheckComb(12, [0, 1, 1])
        self.CheckComb(13, [1, 1, 1])
        self.CheckComb(14, [2, 1, 1])
        self.CheckComb(15, [0, 2, 1])
        self.CheckComb(16, [1, 2, 1])
        self.CheckComb(17, [2, 2, 1])

        self.CheckComb(18, [0, 0, 2])
        self.CheckComb(19, [1, 0, 2])
        self.CheckComb(20, [2, 0, 2])
        self.CheckComb(21, [0, 1, 2])
        self.CheckComb(22, [1, 1, 2])
        self.CheckComb(23, [2, 1, 2])
        self.CheckComb(24, [0, 2, 2])
        self.CheckComb(25, [1, 2, 2])
        self.CheckComb(26, [2, 2, 2])


    def CheckCountCombAllelesEq(self, genos, labels, loci_list, exp_count):
        exp_count = np.array(exp_count)
        res = Libs.BEAM.CountCombAlleles(genos, labels, loci_list)
        self.assertEqual(res.shape, exp_count.shape)
        self.assertTrue(not (res - exp_count).any())


    def test_CountCombAlleles(self):
        self.CheckCountCombAllelesEq([
            [ 0, 0, 0, ],
            [ 0, 0, 0, ],
            [ 0, 0, 0, ],
            [ 0, 0, 0, ],
        ], [ 1, 1, 0, 0, ], [ 0, 1, 2, ], [
            [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
        ])



if __name__ == '__main__':
    unittest.main()

