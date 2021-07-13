import os
import unittest


import Libs.GenosFileFormats



class ReadGenotypeFileUnitTests(unittest.TestCase):
    def test_ReadGenotypeFile(self):
        str_genos_path = os.path.join('Dataset', 'genos_K2_N50_L100_D50.str')
        str_genos = Libs.GenosFileFormats.ReadSTRUCTURE(str_genos_path)

        my_genos_path = os.path.join('Dataset', 'genos_K2_N50_L100_D50.txt')
        my_genos, lbls = Libs.GenosFileFormats.ReadMySTRUCTURE(my_genos_path, '')

        self.assertEqual(len(my_genos), len(str_genos))
        for i in range(len(my_genos)):
            self.assertEqual(len(my_genos[i]), len(str_genos[i]))
            for l in range(len(my_genos[i])):
                self.assertEqual(my_genos[i][l], str_genos[i][l])



if __name__ == '__main__':
    unittest.main()

