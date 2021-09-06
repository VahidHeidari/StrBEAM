import unittest

import numpy as np



EPS = 1e-15



# p  : in probability space, will be clipped into -> 0 <= p <= 1
# lp : in log space or probability space
# Result : in log space
def GetMixLP(lp, p):
    if p <= 0:
        return 0
    elif p >= 1.:
        return lp
    elif lp > 0:
        return np.log((1. - p) * np.exp(-lp) + p) + lp
    else:
        #cp = (1.0 - p)
        #exp_lp = np.exp(lp)
        #l = cp + p * exp_lp
        #res = np.log(cp + (p * exp_lp))
        #print(cp, exp_lp, l, res)
        #return res
        return np.log((1. - p) + p * np.exp(lp))



def MyGetMixLP(lp, p):
    p = max(0.0, min(p, 1.0))       # Clip p into [0, 1]
    if lp > 0:
        return np.log((1.0 - p) * np.exp(-lp) + p) + lp

    return np.log((1.0 - p) + p * np.exp(lp))



def MyGetMixLP2(lp, p):
    p = max(0.0, min(p, 1.0))       # Clip p into [0, 1]
    if lp > 0:
        q = np.exp(-lp)
        return np.log((1.0 - p) + p / q)

    q = np.exp(lp)
    return np.log((1.0 - p) + p * q)



class MixLPTestCase(unittest.TestCase):
    def PrintMixLP(self, lp, p):
        m = GetMixLP(lp, p)
        mm = MyGetMixLP(lp, p)
        m2 = MyGetMixLP2(lp, p)
        if abs(m - mm) > EPS or abs(m - m2) > EPS or abs(mm - m2) > EPS:
            print('       GetMixLP(lp:{}, p:{})  = {}'.format(lp, p, m))
            print('     MyGetMixLP(lp:{}, p:{})  = {}'.format(lp, p, mm))
            print('    MyGetMixLP2(lp:{}, p:{})  = {}'.format(lp, p, m2))
            print('exp(   GetMixLP(lp:{}, p:{})) = {}'.format(lp, p, np.exp(m)))
            print('exp( MyGetMixLP(lp:{}, p:{})) = {}'.format(lp, p, np.exp(mm)))
            print('exp(MyGetMixLP2(lp:{}, p:{})) = {}'.format(lp, p, np.exp(m2)))
            print('')
        self.assertFalse(abs(m - mm) > EPS or abs(m - m2) > EPS or abs(mm - m2) > EPS)


    def test_MixLP(self):
        print('Test 1')
        self.PrintMixLP(0.2, 1.5)
        self.PrintMixLP(0.2, 1)
        self.PrintMixLP(0.2, 0.5)
        self.PrintMixLP(0.2, 0)
        self.PrintMixLP(0.2, -0.5)
        print('--------------------')

        print('Test 2')
        self.PrintMixLP(np.log(0.2), 0.5)
        self.PrintMixLP(-np.log(0.2), 0.5)
        self.PrintMixLP(-1, 0.2)
        self.PrintMixLP(1, 0.2)
        print('--------------------')


if __name__ == '__main__':
    unittest.main()

