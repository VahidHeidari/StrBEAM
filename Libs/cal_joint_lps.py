import unittest

import numpy as np

import common



# Values calculated by original BEAM3 source code
mAlpha = [ -74.546372788379358, -85.226686841512077, -80.308489547462131, -60.188167648307171 ]
mBeta  = [ -42.391297983281106, -41.131126129267471, -34.123379267040661, -17.527794014278896 ]

# Test data set 4
dataC = [
    [ 2, 1, 2, 1, ], [ 0, 1, 2, 1, ], [ 1, 2, 2, 1, ], [ 2, 2, 2, 1, ],
    [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ],
    [ 2, 2, 2, 1, ], [ 2, 2, 1, 1, ], [ 2, 1, 2, 1, ], [ 0, 1, 2, 1, ],
    [ 1, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ],
    [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 1, 1, ],
    [ 2, 1, 2, 1, ], [ 0, 1, 2, 1, ], [ 1, 2, 2, 1, ], [ 2, 2, 2, 1, ],
    [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ],
    [ 2, 2, 2, 1, ], [ 2, 2, 1, 1, ], [ 2, 1, 2, 0, ], [ 0, 1, 2, 1, ],
    [ 1, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ],
    [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 2, 1, ], [ 2, 2, 1, 1, ],
]
dataU = [
    [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ],
    [ 0, 0, 1, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ],
    [ 0, 1, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ],
    [ 0, 0, 0, 0, ], [ 1, 0, 0, 0, ], [ 0, 0, 1, 1, ], [ 0, 0, 0, 0, ],
    [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 1, 0, 0, ], [ 0, 0, 0, 0, ],
    [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 2, 0, 0, 0, ],
    [ 0, 0, 1, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ],
    [ 0, 1, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ],
    [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 0, 1, 0, ], [ 0, 0, 0, 0, ],
    [ 0, 0, 0, 0, ], [ 0, 0, 0, 0, ], [ 0, 1, 0, 0, ], [ 0, 0, 0, 0, ],
]



def GetLogProb(cnt, k):
    return common.GetLogProbDirCat(cnt, k)



# P({g1, g2} | G, Y) = P(g1) x P(g2) x [ P(g1 + g2) / P(g1) P(g2) ]
def CalJointLPS(dataC, dataU, g1, g2):
    g12 = g1 + g2                                                               # Make super node.

    c1  = [ common.CountCombinations(dataC,  g1), common.CountCombinations(dataU,  g1) ]        # Count combinations.
    c2  = [ common.CountCombinations(dataC,  g2), common.CountCombinations(dataU,  g2) ]
    c12 = [ common.CountCombinations(dataC, g12), common.CountCombinations(dataU, g12) ]
    sum_c1 = np.sum(c1, 0)
    sum_c2 = np.sum(c2, 0)
    sum_c12 = np.sum(c12, 0)

    rt_c  = GetLogProb( c12[0], len(g12)) - GetLogProb( c1[0], len(g1)) - GetLogProb( c2[0], len(g2))
    rt_u  = GetLogProb( c12[1], len(g12)) - GetLogProb( c1[1], len(g1)) - GetLogProb( c2[1], len(g2))
    rt_cu = GetLogProb(sum_c12, len(g12)) - GetLogProb(sum_c1, len(g1)) - GetLogProb(sum_c2, len(g2))

    return rt_c, rt_u, rt_cu



class CalJoinLPSTestCase(unittest.TestCase):
    def test_CalJoinLP(self):
        g1 = [ 0, 1 ]
        g2 = [ 2, 3 ]
        rt_c, rt_u, rt_cu = CalJointLPS(dataC, dataU, g1, g2)
        print('rt.c  : {}'.format(rt_c))
        print('rt.u  : {}'.format(rt_u))
        print('rt.cu : {}'.format(rt_cu))
        self.assertFalse(abs(-3.97567 - rt_c) > 1e-5)
        self.assertFalse(abs(-6.74916 - rt_u) > 1e-5)
        self.assertFalse(abs(30.88588 - rt_cu) > 1e-5)


if __name__ == '__main__':
    unittest.main()

