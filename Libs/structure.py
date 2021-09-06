import os

import numpy as np
import scipy.stats as st

import confs
import priors



def InitializeParameters(genos, num_clusters):
    num_indivs = len(genos)
    num_loci = len(genos[0])

    z = np.random.random_integers(0, num_clusters - 1, num_indivs * num_loci)
    z = z.reshape(num_indivs, num_loci)

    q = np.zeros((num_indivs, num_clusters), dtype=np.float)

    p = np.ones((num_clusters, num_loci, priors.NUM_ALLELES)) / priors.NUM_ALLELES
    p += (np.random.rand(num_clusters, num_loci, priors.NUM_ALLELES) - 0.5) * priors.ALLELE_PRIOR
    rp = p.sum(2).repeat(priors.NUM_ALLELES, axis=1)
    p /= rp.reshape(num_clusters, num_loci, priors.NUM_ALLELES)
    return z, q, p


def InitializeSamples(genos, num_clusters):
    num_indivs = len(genos)
    num_loci = len(genos[0])
    smpl_z = np.zeros((num_indivs, num_loci, num_clusters), dtype=np.float)
    smpl_q = np.zeros((num_indivs, num_clusters), dtype=np.float)
    smpl_p = np.zeros((num_clusters, num_loci, priors.NUM_ALLELES), dtype=np.float)
    return smpl_z, smpl_q, smpl_p


def CountAllelesAndOrigins(genos, z, num_clusters):
    num_indivs = len(genos)
    num_loci = len(genos[0])
    cnt_genos = np.zeros((num_clusters, num_loci, priors.NUM_ALLELES), dtype=np.int)
    cnt_origs = np.zeros((num_indivs, num_clusters), dtype=np.int)

    # Count alleles and sub-population of origins.
    for i in range(num_indivs):
        for l in range(num_loci):
            k, g = z[i, l], genos[i][l]
            cnt_genos[k, l, g] += 1
            cnt_origs[i, k] += 1
    return cnt_genos, cnt_origs


def DrawSampleZ(lg_probs):
    mx_probs = lg_probs.max()
    sum_probs = mx_probs + np.log(np.sum(np.exp(lg_probs - mx_probs)))
    probs = np.exp(lg_probs - sum_probs)
    rvs_z = st.multinomial.rvs(1, probs).argmax()
    return rvs_z


def CalculateLikelihood(genos, z, q, p):
    llhood = 0
    for i in range(len(genos)):
        for l in range(len(genos[0])):
            k, g = z[i, l], genos[i][l]
            llhood += np.log(p[k, l, g])
    return llhood


def WriteParamsToFile(base_path, genos_path, smpl_z, smpl_q, smpl_p, cnt_smpls):
    num_indivs = len(smpl_z)
    num_loci = len(smpl_z[0])
    num_clusters = len(smpl_q[0])

    smpl_z /= cnt_smpls
    base_name = os.path.basename(genos_path)
    z_path = os.path.join(base_path, 'smpl_z-{}.txt').format(base_name)
    with open(z_path, 'w') as f:
        for i in range(num_indivs):
            for l in range(num_loci):
                f.write('  '.join(['%0.02f' % smpl_z[i, l, k] for k in range(num_clusters)]) + '\n')

    smpl_q /= cnt_smpls
    q_path = os.path.join(base_path, 'smpl_q-{}.txt').format(base_name)
    with open(q_path, 'w') as f:
        for i in range(num_indivs):
            f.write('%3d    ' % smpl_q[i].argmax())
            f.write('  '.join(['%0.02f' % smpl_q[i, k] for k in range(num_clusters)]) + '\n')

    smpl_p /= cnt_smpls
    p_path = os.path.join(base_path, 'smpl_p-{}.txt').format(base_name)
    with open(p_path, 'w') as f:
        for k in range(num_clusters):
            for l in range(num_loci):
                f.write('  '.join(['%0.02f' % smpl_p[k, l, a] for a in range(priors.NUM_ALLELES)]) + '\n')
            f.write('\n')


def IsConverged(diff_llhood, itr, cnt_smpls):
    if itr < confs.BURNIN_ITERS:
        return False
    return diff_llhood < confs.CONV_EPSILONE and cnt_smpls > confs.MIN_SAMPLE_SIZE




if __name__ == '__main__':
    print('This is a module!')

