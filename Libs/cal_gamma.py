import unittest

import numpy as np

import cal_joint_lps
import data_set_4
import mix_lp



def CalGamma(dataC, dataU, pD, pA, g1, g2, calc_type):
    rt_c, rt_u, rt_cu = cal_joint_lps.CalJointLPS(dataC, dataU, g1, g2)

    if calc_type == 0:
        res = mix_lp.MyGetMixLP2(rt_cu, pA)
        return res

    lp = rt_c + mix_lp.MyGetMixLP2(rt_u, pA)
    res = mix_lp.MyGetMixLP2(lp, pD)
    return res



class CalGammaTestCase(unittest.TestCase):
    def test_CalGamma(self):
        dataC = data_set_4.dataC
        dataU = data_set_4.dataU

        # Test _calGamma_S.
        g1 = [0, 1]
        g2 = [2, 3]
        res = CalGamma(dataC, dataU, 0.1, 0.1, g1, g2, 0)
        print('_calGamma_S:')
        print('res with cal_type = 0   ->   {:3.5f}   {}'.format(res, np.exp(res)))
        self.assertFalse(abs(28.58330 - res) > 1e-5)

        res = CalGamma(dataC, dataU, 0.1, 0.1, g1, g2, 1)
        print('res with cal_type = 1   ->   {:3.5f}   {}'.format(res, np.exp(res)))
        self.assertFalse(abs(-0.10349 - res) > 1e-5)
        print('')

        # Test _calGamma_E.
        g1 = [0, 1]
        g2 = [3]            # Exclude `2' from g2
        res = CalGamma(dataC, dataU, 0.1, 0.1, g1, g2, 0)
        print('_calGamma_E:')
        print('res with cal_type = 0   ->   {:3.5f}   {}'.format(res, np.exp(res)))
        self.assertFalse(abs(30.52722 - res) > 1e-5)

        res = CalGamma(dataC, dataU, 0.1, 0.1, g1, g2, 1)
        print('res with cal_type = 1   ->   {:3.5f}   {}'.format(res, np.exp(res)))
        self.assertFalse(abs(-0.0524233 - res) > 1e-5)


if __name__ == '__main__':
    unittest.main()

