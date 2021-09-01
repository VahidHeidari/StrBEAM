import os
import sys
import time

import numpy as np
import scipy.stats as st

import Confs
import Libs.CalcAccuracy
import Libs.GenosFileFormats
import Libs.Logger
import Libs.PrettyPrinting



def ParseCmd(argv):
    if len(argv) < 3:
        print('Usage   ' + os.path.basename(argv[0]) + '   K  INPUT_PATH')
        print('')
        print('  K          : Number of clusters')
        print('  INPUT_PATH : Genotype file path')
        exit(1)

    num_clusters = int(argv[1])
    genos_path = argv[2]
    return num_clusters, genos_path


def InitializeParameters(genos, num_clusters):
    num_indivs = len(genos)
    num_loci = len(genos[0])

    z = np.random.random_integers(0, num_clusters - 1, num_indivs * num_loci)
    z = z.reshape(num_indivs, num_loci)

    q = np.zeros((num_indivs, num_clusters), dtype=np.float)

    p = np.ones((num_clusters, num_loci, Confs.NUM_ALLELES)) / Confs.NUM_ALLELES
    p += (np.random.rand(num_clusters, num_loci, Confs.NUM_ALLELES) - 0.5) * Confs.ALLELE_PRIOR
    rp = p.sum(2).repeat(Confs.NUM_ALLELES, axis=1)
    p /= rp.reshape(num_clusters, num_loci, Confs.NUM_ALLELES)
    return z, q, p


def InitializeSamples(genos, num_clusters):
    num_indivs = len(genos)
    num_loci = len(genos[0])
    smpl_z = np.zeros((num_indivs, num_loci, num_clusters), dtype=np.float)
    smpl_q = np.zeros((num_indivs, num_clusters), dtype=np.float)
    smpl_p = np.zeros((num_clusters, num_loci, Confs.NUM_ALLELES), dtype=np.float)
    return smpl_z, smpl_q, smpl_p


def CountAllelesAndOrigins(genos, z, num_clusters):
    num_indivs = len(genos)
    num_loci = len(genos[0])
    cnt_genos = np.zeros((num_clusters, num_loci, Confs.NUM_ALLELES), dtype=np.int)
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
                f.write('  '.join(['%0.02f' % smpl_p[k, l, a] for a in range(Confs.NUM_ALLELES)]) + '\n')
            f.write('\n')


def IsConverged(diff_llhood, itr, cnt_smpls):
    if itr < Confs.BURNIN_ITERS:
        return False
    return diff_llhood < Confs.CONV_EPSILONE and cnt_smpls > Confs.MIN_SAMPLE_SIZE



if __name__ == '__main__':
    Libs.Logger.Log('\n\n{}  Running Structure . . .'.format(time.ctime()))
    num_clusters, genos_path = ParseCmd(sys.argv)
    genos = Libs.GenosFileFormats.ReadGenotypeFile(genos_path)
    if len(genos) == 0:
        Libs.Logger.Log('Could not read genotype file!')
        exit(2)

    num_indivs = len(genos)
    num_loci = len(genos[0])
    Libs.Logger.Log('  K            : %d' % num_clusters)
    Libs.Logger.Log('  Input Path   : %s' % genos_path)
    Libs.Logger.Log('  Num Indivs   : %d' % num_indivs)
    Libs.Logger.Log('  Num Loci     : %d' % num_loci)
    Libs.Logger.Log('  BURNIN       : %d' % Confs.BURNIN_ITERS)
    Libs.Logger.Log('  MCMC         : %d' % Confs.MAX_ITERS)
    Libs.Logger.Log('  Log Iter     : %d' % Confs.LOG_ITERS)
    Libs.Logger.Log('  Thinning     : %d' % Confs.THINNING)
    Libs.Logger.Log('  Conv Epsilon : %f' % Confs.CONV_EPSILONE)
    Libs.Logger.Log('\ngenos:')
    Libs.PrettyPrinting.PrintListOfLists(genos)

    Libs.Logger.Log('\n{}  Initialize parameters . . .'.format(time.ctime()))
    z, q, p = InitializeParameters(genos, num_clusters)
    prior_lambda = np.ones((num_clusters, Confs.NUM_ALLELES), dtype=np.float) * 0.1
    prior_alpha = np.ones((num_clusters), dtype=np.float) / num_clusters
    cnt_genos, cnt_origs = CountAllelesAndOrigins(genos, z, num_clusters)

    smpl_z, smpl_q, smpl_p = InitializeSamples(genos, num_clusters)
    cnt_smpls = 0

    Libs.Logger.Log('{}  Start of MCMC . . .'.format(time.ctime()))
    cnt_thining = Confs.THINNING
    llhood = CalculateLikelihood(genos, z, q, p)
    diff_llhood = 0
    for itr in range(1, Confs.BURNIN_ITERS + Confs.MAX_ITERS + 1):              # Run MCMC sampling loop.
        # Log progress.
        if itr == 1 or (itr % Confs.LOG_ITERS) == 0:
            loop_str = 'BURNIN' if itr < Confs.BURNIN_ITERS else 'MCMC'
            fmt_tuple = (time.ctime(), loop_str, itr, llhood, diff_llhood, cnt_smpls)
            Libs.Logger.Log('%s  %s  itr #%d    llhood -> %f   diff -> %f    #smpls -> %d' % fmt_tuple)

        # Update P.
        for l in range(num_loci):
            tmp_sm = prior_lambda + cnt_genos[:, l, :]
            for k in range(num_clusters):
                p[k, l, :] = st.dirichlet.rvs(tmp_sm[k])

        # Update Q and Z.
        for i in range(num_indivs):
            q[i, :] = st.dirichlet.rvs(prior_alpha + cnt_origs[i, :])
            lg_q = np.log(q[i, :])
            for l in range(num_loci):
                g = genos[i][l]
                lg_p = np.log(p[:, l, g])
                old_z = z[i, l]
                new_z = z[i, l] = DrawSampleZ(lg_q + lg_p)
                if old_z != new_z:
                    cnt_genos[old_z, l, g] -= 1
                    cnt_genos[new_z, l, g] += 1
                    cnt_origs[i, old_z] -= 1
                    cnt_origs[i, new_z] += 1

        # Save samples if we are in MCMC loop.
        if itr > Confs.BURNIN_ITERS:
            cnt_thining += 1
            if cnt_thining > max(1, Confs.THINNING):
                cnt_thining = 1
                cnt_smpls += 1
                smpl_p += p
                smpl_q += q
                for i in range(num_indivs):
                    for l in range(num_loci):
                        k = z[i, l]
                        smpl_z[i, l, k] += 1

        # Calculate log likelihood.
        new_llhood = CalculateLikelihood(genos, z, q, p)
        diff_llhood = abs(new_llhood - llhood)
        llhood = new_llhood

        # Check convergence.
        if IsConverged(diff_llhood, itr, cnt_smpls):
            fmt_tuple = (time.ctime(), itr, llhood, diff_llhood, cnt_smpls)
            Libs.Logger.Log('%s  Break itr #%d    llhood -> %f    diff -> %f #smpls -> %d' % fmt_tuple)
            break

    Libs.Logger.Log('%s  End of MCMC!' % time.ctime())
    Libs.Logger.Log('  Num Samples : %d' % cnt_smpls)

    # Write samples to files.
    WriteParamsToFile(Libs.Logger.LOG_OUT_DIR, genos_path, smpl_z, smpl_q, smpl_p, cnt_smpls)

    # Calculate accuracy of clustering.
    if num_clusters < 6:
        clusters = [ smpl_q[i].argmax() for i in range(num_indivs) ]
        clusters_cnt = Libs.CalcAccuracy.GetClusterCounts(num_clusters, clusters)
        accs = Libs.CalcAccuracy.CalcAccuracy(clusters_cnt)
        mx_acc = int(max(accs) * 100.0)
        Libs.Logger.Log('  Acc : %d%%   %s' % (mx_acc, str(np.round(accs, 2))))
    else:
        Libs.Logger.Log('  Acc : 0')

