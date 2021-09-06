import copy
import os
import random
import sys

import numpy as np
import scipy.stats as st

import Libs.confs
import Libs.genos_file_formats
import Libs.logger
import Libs.pretty_printing
import Libs.priors



def ParseCmd(argv):
    if len(argv) < 5:
        print('Usage   ' + os.path.basename(argv[0]) + '   K  N  L  D  [--disease]')
        print('')
        print('  K         : Number of clusters')
        print('  N         : Number of individuals in each cluster')
        print('  L         : Number of loci')
        print('  D         : Percent of loci with different MAF')
        print('  --disease : Make disease labels (default is false)')
        exit(1)

    num_clusters = int(argv[1])
    num_indivs = int(argv[2])
    num_loci = int(argv[3])
    diff_maf = int(argv[4])

    is_disease = False
    if '--disease' in argv:
        is_disease = True

    Libs.logger.Log(' K          : ' + str(num_clusters))
    Libs.logger.Log(' N          : ' + str(num_indivs))
    Libs.logger.Log(' L          : ' + str(num_loci))
    Libs.logger.Log(' D          : ' + str(diff_maf))
    Libs.logger.Log(' Is Disease : ' + str(is_disease))
    return num_clusters, num_indivs, num_loci, diff_maf, is_disease


def SelectLociDiffMAFs(num_loci, diff_mafs_percent):
    loci_list = []
    num_diff_loci = int(num_loci / 100.0 * diff_mafs_percent)
    Libs.logger.Log('Num Diff Loci : %d' % num_diff_loci)
    num_diff_loci = max(1, int(num_diff_loci + 0.5))
    while len(loci_list) < num_diff_loci:
        l = int(random.uniform(0, num_loci))
        if l not in loci_list:
            loci_list.append(l)
    loci_list = sorted(loci_list)
    return loci_list


def IsDuplicated(loci, loci_list):
    for k in range(len(loci_list)):
        if loci in loci_list[k]:
            return True
    return False


def SelectDiseaseLociAndSNPs(num_loci, num_clusters, diff_loci):
    loci_list = [ [] for k in range(num_clusters) ]
    for k in range(num_clusters):
        while len(loci_list[k]) < 3:
            l = int(random.uniform(0, num_loci))
            if not IsDuplicated(l, loci_list):
                loci_list[k].append(l)
        loci_list[k] = sorted(loci_list[k])
    snp_list = [ [ int(random.uniform(0, Libs.priors.NUM_ALLELES)) for i in range(3) ] for k in range(num_clusters) ]
    return loci_list, snp_list


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


def MakeRandomFreqs(num_clusters, num_loci, diff_loci):
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
        return MakeRandomFreqs(num_clusters, num_loci, diff_loci)
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


def CalcDiseaseStatus(genotype, cluster, dise_loci, snp_list, is_bernoulli=False):
    snps = [ genotype[l] for l in dise_loci[cluster] ]
    indic = [ 1 if snps[l] == snp_list[cluster][l] else 0 for l in range(len(snps)) ]
    z = 1.5 * np.prod(indic)
    prob = 1.0 / (1.0 + np.exp(-z))
    y = 0 if prob <= 0.5 else 1
    if is_bernoulli:
        y = st.bernoulli.rvs(prob)
    return y == 1


def WriteToFile(genos, labels, diff_loci, freqs, dise_loci, snp_list, params_tuple):
    num_clusters = params_tuple[0]
    if not os.path.isdir('Dataset'):
        os.makedirs('Dataset')

    # Write compressed file.
    str_path = os.path.join('Dataset', 'genos_K%d_N%d_L%d_D%d.str' % params_tuple)
    Libs.genos_file_formats.WriteSTRUCTURE(str_path, genos)

    # Write STRUCTURE formatted genotypes.
    my_str_path = os.path.join('Dataset', 'genos_K%d_N%d_L%d_D%d.txt' % params_tuple )
    Libs.genos_file_formats.WriteMySTRUCTURE(my_str_path, genos, lbls, num_clusters)

    # Write allele frequencies.
    freq_path = os.path.join('Dataset', 'freqs_K%d_N%d_L%d_D%d.txt' % params_tuple)
    with open(freq_path, 'w') as f:
        f.write('\n'.join([ str(f) for f in freqs ]))

    # Write loci with difference in MAF.
    diff_path = os.path.join('Dataset', 'diffs_K%d_N%d_L%d_D%d.txt' % params_tuple)
    with open(diff_path, 'w') as f:
        f.write(str(diff_loci))

    if len(lbls) != 0:
        # Write BEAM formatted genotypes.
        beam_path = os.path.join('Dataset', 'genos_beam_K%d_N%d_L%d_D%d.txt' % params_tuple)
        Libs.genos_file_formats.WriteBEAM(beam_path, genos, lbls)

    if len(dise_loci) != 0 and len(snp_list) != 0:
        # Write disease affected loci and corresponding SNPs.
        dise_path = os.path.join('Dataset', 'dise_K%d_N%d_L%d_D%d.txt' % params_tuple)
        with open(dise_path, 'w') as f:
            for k in range(num_clusters):
                f.write(' '.join([ str(d) for d in dise_loci[k] ]) + '\n')
            f.write('\n')
            for k in range(num_clusters):
                f.write(' '.join([ str(s) for s in snp_list[k] ]) + '\n')



if __name__ == '__main__':
    Libs.logger.Log('\n\nStart of mk_dataset.py')
    NUM_CLUSTERS, NUM_INDIVS, NUM_LOCI, DIFF_MAF, IS_DISEASE = ParseCmd(sys.argv)

    diff_loci = SelectLociDiffMAFs(NUM_LOCI, DIFF_MAF)
    Libs.logger.Log('\ndiff loci:')
    Libs.pretty_printing.PrintList(diff_loci)

    freqs = MakeFreqs(NUM_CLUSTERS, NUM_LOCI, diff_loci, True)
    Libs.logger.Log('\nfreqs:')
    Libs.pretty_printing.PrintListOfLists(freqs)

    dise_loci = []
    snp_list = []
    genos = []
    lbls = []
    if IS_DISEASE:
        dise_loci, snp_list = SelectDiseaseLociAndSNPs(NUM_LOCI, NUM_CLUSTERS, diff_loci)
        Libs.logger.Log('\ndisease loci:')
        Libs.pretty_printing.PrintListOfLists(dise_loci)
        Libs.logger.Log('\ndisease SNPs:')
        Libs.pretty_printing.PrintListOfLists(snp_list)

    for k in range(NUM_CLUSTERS):
        if IS_DISEASE:
            cases = []
            ctrls = []
            while len(cases) + len(ctrls) < NUM_INDIVS:
                g = DrawGenos(freqs[k])
                y = CalcDiseaseStatus(g, k, dise_loci, snp_list)
                if y == True:
                    if len(cases) < NUM_INDIVS // 2:
                        cases.append(g)
                else:
                    if len(ctrls) < NUM_INDIVS // 2 + (0 if NUM_INDIVS % 2 == 0 else 1):
                        ctrls.append(g)
            genos += ctrls + cases
            lbls += [ 0 for i in range(len(ctrls)) ] + [ 1 for i in range(len(cases)) ]
        else:
            genos += [ DrawGenos(freqs[k]) for n in range(NUM_INDIVS) ]

    Libs.logger.Log('\ngenos:')
    Libs.pretty_printing.PrintListOfLists(genos)
    if IS_DISEASE:
        Libs.logger.Log('\nlabels:')
        Libs.pretty_printing.PrintList(lbls)

    params_tuple = (NUM_CLUSTERS, NUM_INDIVS, NUM_LOCI, DIFF_MAF)
    WriteToFile(genos, lbls, diff_loci, freqs, dise_loci, snp_list, params_tuple)

