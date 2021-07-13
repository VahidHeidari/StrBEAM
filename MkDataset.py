import copy
import os
import random
import sys

import Libs.GenosFileFormats
import Libs.Logger
import Libs.PrettyPrinting



def ParseCmd(argv):
    if len(argv) < 5:
        print('Usage   ' + os.path.basename(argv[0]) + '   K  N  L  D')
        print('')
        print('  K : Number of clusters')
        print('  N : Number of individuals in each cluster')
        print('  L : Number of loci')
        print('  D : Percent of loci with different MAF')
        exit(1)

    num_clusters = int(argv[1])
    num_indivs = int(argv[2])
    num_loci = int(argv[3])
    diff_maf = int(argv[4])

    Libs.Logger.Log(' K: ' + str(num_clusters))
    Libs.Logger.Log(' N: ' + str(num_indivs))
    Libs.Logger.Log(' L: ' + str(num_loci))
    Libs.Logger.Log(' D: ' + str(diff_maf))
    return num_clusters, num_indivs, num_loci, diff_maf


def SelectLociDiffMAFs(num_loci, diff_mafs_percent):
    loci_list = []
    num_diff_loci = int(num_loci / 100.0 * diff_mafs_percent)
    Libs.Logger.Log('Num Diff Loci : %d' % num_diff_loci)
    num_diff_loci = max(1, int(num_diff_loci + 0.5))
    while len(loci_list) < num_diff_loci:
        l = int(random.uniform(0, num_loci))
        if l not in loci_list:
            loci_list.append(l)
    loci_list = sorted(loci_list)
    return loci_list


def MakeRandomBaseFreq(num_loci):
    return [ round(random.uniform(0, 1), 2) for l in range(num_loci) ]


def GetRoundGenoProb():
    GENOS_WHEEL = [ 1.0 / 3, 2.0 / 3, 1.0 ]
    PROBS = [ 0, 0.5, 1 ]
    u = random.uniform(0, 1)
    for g in range(len(GENOS_WHEEL)):
        if u < GENOS_WHEEL[g]:
            return PROBS[g]

    return PROBS[len(PROBS)]


def MakeRoundBaseFreq(num_loci):
    return [ GetRoundGenoProb() for l in range(num_loci) ]


def MakeRoundFreqs(num_clusters, num_loci, diff_loci):
    base_freq = MakeRandomBaseFreq(num_loci)
    freqs = [ copy.deepcopy(base_freq) for k in range(num_clusters) ]
    for k in range(1, num_clusters):
        for l in diff_loci:
            freqs[k][l] = round(random.uniform(0, 1), 2)
    return freqs


def MakeRoundFreqs(num_clusters, num_loci, diff_loci):
    base_freq = MakeRoundBaseFreq(num_loci)
    freqs = [ copy.deepcopy(base_freq) for k in range(num_clusters) ]
    for k in range(1, num_clusters):
        for l in diff_loci:
            freqs[k][l] = GetRoundGenoProb()
    return freqs


def MakeFreqs(num_clusters, num_loci, diff_loci, is_random=False):
    if is_random:
        return MakeRoundFreqs(num_clusters, num_loci, diff_loci)

    return MakeRoundFreqs(num_clusters, num_loci, diff_loci)


def GetGenos(freq):
    u1 = random.uniform(0, 1)
    a1 = 1 if u1 < freq else 0
    u2 = random.uniform(0, 1)
    a2 = 1 if u2 < freq else 0
    return a1 + a2


def DrawGenos(freqs):
    num_loci = len(freqs)
    genos = [ GetGenos(freqs[l]) for l in range(num_loci) ]
    return genos



if __name__ == '__main__':
    Libs.Logger.Log('\n\nStart of MkDataset.py')

    NUM_CLUSTERS, NUM_INDIVS, NUM_LOCI, DIFF_MAF = ParseCmd(sys.argv)
    diff_loci = SelectLociDiffMAFs(NUM_LOCI, DIFF_MAF)
    Libs.Logger.Log('\ndiff loci:')
    Libs.PrettyPrinting.PrintList(diff_loci)

    freqs = MakeFreqs(NUM_CLUSTERS, NUM_LOCI, diff_loci, True)
    Libs.Logger.Log('\nfreqs:')
    Libs.PrettyPrinting.PrintListOfLists(freqs)

    genos = []
    for k in range(NUM_CLUSTERS):
        genos += [ DrawGenos(freqs[k]) for n in range(NUM_INDIVS) ]
    Libs.Logger.Log('\ngenos:')
    Libs.PrettyPrinting.PrintListOfLists(genos)

    PARAMS_TUPLE = (NUM_CLUSTERS, NUM_INDIVS, NUM_LOCI, DIFF_MAF)
    my_str_path = os.path.join('Dataset', 'genos_K%d_N%d_L%d_D%d.txt' % PARAMS_TUPLE )
    str_path = os.path.join('Dataset', 'genos_K%d_N%d_L%d_D%d.str' % PARAMS_TUPLE)
    diff_path = os.path.join('Dataset', 'diffs_K%d_N%d_L%d_D%d.txt' % PARAMS_TUPLE)
    freq_path = os.path.join('Dataset', 'freqs_K%d_N%d_L%d_D%d.txt' % PARAMS_TUPLE)
    if not os.path.isdir('Dataset'):
        os.makedirs('Dataset')
    Libs.GenosFileFormats.WriteMySTRUCTURE(my_str_path, genos, [], NUM_CLUSTERS)
    Libs.GenosFileFormats.WriteSTRUCTURE(str_path, genos)
    with open(diff_path, 'w') as f:
        f.write(str(diff_loci))
    with open(freq_path, 'w') as f:
        f.write('\n'.join([str(f) for f in freqs]))

