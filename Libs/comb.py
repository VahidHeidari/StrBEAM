import copy
import unittest



def GetNextComb(lst, sz):
    j = len(lst) - 1
    lst[j] += 1
    while lst[j] >= sz - len(lst) + 1 + j:
        j -= 1
        if j < 0:
            return False

        lst[j] += 1

    for k in range(j + 1, len(lst)):
        lst[k] = lst[k - 1] + 1
    return True


def MakeAllCombinations(sz, k):
    if k == 0:
        return [[]]

    lst = range(0, min(sz, k))
    cmbs = []
    while True:
        cmbs.append(copy.deepcopy(lst))
        if not GetNextComb(lst, sz):
            break

    return cmbs



class CombinationTestCase(unittest.TestCase):
    def test_Comb0(self):
        combs = MakeAllCombinations(5, 0)
        self.assertEqual(combs, [[]])


    def test_Comb1(self):
        combs = MakeAllCombinations(5, 1)
        self.assertEqual(combs, [[0], [1], [2], [3], [4]])


    def test_Comb2(self):
        combs = MakeAllCombinations(5, 2)
        self.assertEqual(combs, [
            [0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
            [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],
        ])


    def test_Comb3(self):
        combs = MakeAllCombinations(5, 3)
        self.assertEqual(combs, [
            [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4],
            [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4],
        ])


    def test_Comb4(self):
        combs = MakeAllCombinations(5, 4)
        self.assertEqual(combs, [
            [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4],
        ])


    def test_Comb5(self):
        combs = MakeAllCombinations(5, 5)
        self.assertEqual(combs, [[0, 1, 2, 3, 4]])


if __name__ == '__main__':
    unittest.main()

