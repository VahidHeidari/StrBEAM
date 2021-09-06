import os
import unittest

import numpy as np
import scipy.stats as st

import BEAM
import logger



def ReadBEAMWithPositions(in_path):
    if not os.path.isfile(in_path):
        return [], [], []

    lines = [ l.strip() for l in open(in_path, 'r') ]                           # Read file lines.
    lbls = [ int(l) for l in lines[0].split()[3:] ]                             # Read labels.
    snps = [ [ int(l) for l in i.split()[3:] ] for i in lines[1:] ]             # Read SNPs.

    pos = []
    for l in lines[1:]:
        sp = l.split()
        chrom = int(sp[1][3:]) if sp[1].lower().startswith('chr')           \
                else int(sp[1][2:]) if sp[1].lower().startswith('ch')       \
                else sp[1]
        pos.append((chrom, int(sp[2])))

    num_loci = len(snps)                                                        # Transpose snps.
    num_indivs = len(lbls)
    genos = [ [ snps[l][i] for l in range(num_loci) ] for i in range(num_indivs) ]
    return genos, lbls, pos


def ReadBEAM(in_path):
    genos, lbls, pos = ReadBEAMWithPositions(in_path)
    return gneos, lbls


def WriteBEAM(out_path, genos, labels):
    num_indivs = len(labels)
    if len(genos) != num_indivs:                                                # Check consistency.
        print('WARNING: Num indivs in genos (' +
                str(len(genos)) + ') is not equat to num indivs in labels (' +
                str(num_indivs) + ')!' +
                '\n         It will be set to minimum number.')
        num_indivs = min(len(genos), num_indivs)

    num_loci = len(genos[0])
    with open(out_path, 'w') as f:                                              # Write into BEAM file format.
        f.write('ID Chr Pos ' + ' '.join([ str(l) for l in labels ]) + '\n')    # Write labels.

        f.write('rs{} Chr1 {} '.format(1, 1))                                   # Write SNPs.
        f.write(' '.join([ str(genos[i][0]) for i in range(num_indivs) ]))
        for l in range(1, num_loci):
            f.write('\nrs{} Chr1 {} '.format(l + 1, l + 1))
            f.write(' '.join([str(genos[i][l]) for i in range(num_indivs)]))


def ReadMySTRUCTURE(geno_path, label_path):
    with open(geno_path, 'r') as f:                                             # Read individuals.
        num_indivs = int(f.readline().split(':')[1].strip())
        num_loci = int(f.readline().split(':')[1].strip())
        num_clusters = int(f.readline().split(':')[1].strip())
        f.readline()
        total_indivs = num_indivs * num_clusters
        indivs = [ [ int(g) for g in f.readline().strip() ] for i in range(total_indivs) ]

    labels = []
    if len(label_path):
        with open(label_path, 'r') as f:                                        # Read labels.
            labels = [ 1 if f.readline().strip() == 'YES' else 0 for i in range(total_indivs) ]
    return indivs, labels


def WriteMySTRUCTURE(out_path, genos, labels, num_clusters):
    num_indivs = len(genos)
    if len(genos) != num_indivs and len(labels) != 0:                           # Check consistency.
        print('WARNING: Num indivs in genos (' +
                str(len(genos)) + ') is not equat to num indivs in labels (' +
                str(num_indivs) + ')!' +
                '\n         It will be set to minimum number.')
        num_indivs = min(len(genos), num_indivs)

    num_loci = len(genos[0])
    with open(out_path, 'w') as f:                                              # Write into BEAM file format.
        f.write('NUM_INDIVS: {}\n'.format(num_indivs // num_clusters))
        f.write('NUM_LOCI: {}\n'.format(num_loci))
        f.write('NUM_CLUSTERS: {}\n\n'.format(num_clusters))

        for n in range(num_indivs):
            f.write(''.join([ str(g) for g in genos[n] ]))
            if n + 1 < num_indivs:
                f.write('\n')

    if len(labels):
        base_name = os.path.splitext(os.path.basename(out_path))[0]
        base_dir = os.path.dirname(out_path)
        lbl_out_path = os.path.join(base_dir, base_name + '-labels.txt')
        with open(lbl_out_path, 'w') as f:
            for n in range(num_indivs):
                f.write('YES' if labels[n] == 1 else 'NO')
                if n + 1 < num_indivs:
                    f.write('\n')


def ReadSTRUCTURE(geno_path):
    with open(geno_path, 'r') as f:
        indivs = []
        chrom = 0
        for l in f:
            sp = l.strip().split(' ')
            if chrom == 0:
                indivs.append([ int(g) for g in sp[1:] ])
            else:
                for i in range(len(indivs[-1])):
                    indivs[-1][i] += int(sp[i + 1])
            chrom = (chrom + 1) % 2
    return indivs


def WriteSTRUCTURE(out_path, genos):
    with open(out_path, 'w') as f:
        for n in range(len(genos)):
            f.write('IND_{} {}\n'.format(n + 1, ' '.join([str(1 if g >= 1 else 0) for g in genos[n]])))
            f.write('IND_{} {}'.format(n + 1, ' '.join([str(1 if g == 2 else 0) for g in genos[n]])))
            if n + 1 < len(genos):
                f.write('\n')


def ReadGenotypeFile(genos_path):
    if not os.path.isfile(genos_path):
        return []
    with open(genos_path, 'r') as f:
        line = f.readline()
    if line.startswith('IND_'):
        return ReadSTRUCTURE(genos_path)
    if line.startswith('NUM_INDIVS:'):
        genos, lbls = ReadMySTRUCTURE(genos_path, '')
        return genos
    return []


def FillMissings(genos):
    num_indivs = len(genos)
    if num_indivs == 0:
        return genos

    num_loci = len(genos[0])
    if num_loci == 0:
        return genos

    for l in range(num_loci):
        col = np.array(genos)[:, l]
        gn, cnt = np.unique(col, return_counts=True)
        freqs = np.zeros(4)
        for i in range(len(gn)):
            freqs[i if 0 <= i < 3 else 3] += cnt[i]
        freqs[:3] /= freqs[:3].sum()
        for n in range(num_indivs):
            if not 0 <= genos[n][l] < 3:
                genos[n][l] = st.multinomial.rvs(1, freqs[:3], 1).tolist()[0].index(1)
    return genos



class DataGenUnitTests(unittest.TestCase):
    def TestProbCatDir(self, count1, count2, prior):
        logger.Log((count1, count2, prior))
        r1 = BEAM.GetLogMarginalCatDir(count1, prior)
        r2 = BEAM.GetLogMarginalCatDir(count2, prior)
        fmt = 'r1:{}    r2:{}\n' +                                              \
              '   exp(r1)={}    exp(r2)={}\n' +                                 \
              '   r1 < r2 {}\n' +                                               \
              '   r1 / r2 {}\n'
        logger.Log(fmt.format(r1, r2, np.exp(r1), np.exp(r2), r1 < r2, np.exp(r1 - r2)))


    def TestProbIndepCatDir(self, count1, count2, prior):
        r1 = BEAM.GetLogMarginalCatDir(count1, prior)
        r2 = BEAM.GetLogMarginalCatDir(count2, prior)
        count12 = np.sum([count1, count2], 0).tolist()
        r3 = BEAM.GetLogMarginalCatDir(count12, prior)
        fmt = 'count1:{} count2:{} prior:{}\n' +                                \
              '   r1:{}    r2:{}   r1+r2:{}    r3:{}\n' +                       \
              '   exp(r1)={}    exp(r2)={}    exp(r1 + r2)={}    exp(r3)={}\n' +\
              '   Disease Locus (r3 < r1 + r2) ? {}\n'
        logger.Log(fmt.format(count1, count2, prior,
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
        res = BEAM.GetComb(cmb)
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
        res = BEAM.CountCombAlleles(genos, labels, loci_list)
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


    def test_ReadGenotypeFile(self):
        str_genos_path = os.path.join('Dataset', 'genos_K2_N50_L100_D50.str')
        str_genos = ReadSTRUCTURE(str_genos_path)

        my_genos_path = os.path.join('Dataset', 'genos_K2_N50_L100_D50.txt')
        my_genos, lbls = ReadMySTRUCTURE(my_genos_path, '')

        self.assertEqual(len(my_genos), len(str_genos))
        for i in range(len(my_genos)):
            self.assertEqual(len(my_genos[i]), len(str_genos[i]))
            for l in range(len(my_genos[i])):
                self.assertEqual(my_genos[i][l], str_genos[i][l])


if __name__ == '__main__':
    unittest.main()

