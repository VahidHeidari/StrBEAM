import os
import sys
import random
import time

import numpy as np
import scipy.stats as st

import Libs.cal_full_lp
import Libs.cal_graph_lp
import Libs.calc_accuracy
import Libs.calc_alpha_beta
import Libs.confs
import Libs.disease_graph
import Libs.genos_file_formats
import Libs.initialize_marker_member
import Libs.logger
import Libs.outputs
import Libs.pretty_printing
import Libs.priors
import Libs.structure
import Libs.switch_marker_member
import Libs.update_marker_member



def ParseCmd(argv):
    if len(argv) < 3:
        print('Usage   ' + os.path.basename(argv[0]) + '    K  INPUT_PATH  [--verbose]')
        print('')
        print('  K          : Number of clusters')
        print('  INPUT_PATH : Genotype file path in BEAM compatible format')
        print('  --verbose  : Print and dump all outputs')
        exit(1)

    num_clusters = int(argv[1])
    genos_path = argv[2]
    is_verbose = True if '--verbose' in argv else False
    return num_clusters, genos_path, is_verbose



if __name__ == '__main__':
    Libs.logger.Log('\n\n{}  Running StrBEAM . . .'.format(time.ctime()))
    num_clusters, genos_path, is_verbose = ParseCmd(sys.argv)

    # Read genotypes and labels.
    genos, labels, positions = Libs.genos_file_formats.ReadBEAMWithPositions(genos_path)
    if len(genos) == 0 or len(labels) == 0:
        Libs.logger.Log('Could not read geno type file!')
        exit(2)

    num_indivs = len(genos)
    num_lbls = len(labels)
    if num_indivs != num_lbls:
        Libs.logger.Log('Inconsistency in num indivs(%d) and num labels(%d)!' % (num_indivs, num_lbls))
        exit(3)

    num_loci = len(genos[0])
    Libs.logger.Log('  K            : %d' % num_clusters)
    Libs.logger.Log('  Input Path   : %s' % genos_path)
    Libs.logger.Log('  Num Indivs   : %d' % num_indivs)
    Libs.logger.Log('  Num Loci     : %d' % num_loci)
    Libs.logger.Log('  BURNIN       : %d' % Libs.confs.BURNIN_ITERS)
    Libs.logger.Log('  MCMC         : %d' % Libs.confs.MAX_ITERS)
    Libs.logger.Log('  Log Iter     : %d' % Libs.confs.LOG_ITERS)
    Libs.logger.Log('  Thinning     : %d' % Libs.confs.THINNING)
    Libs.logger.Log('  Conv Epsilon : %f' % Libs.confs.CONV_EPSILONE)
    Libs.logger.Log('\ngenos:')
    Libs.pretty_printing.PrintListOfLists(genos)
    Libs.logger.Log('\n')

    Libs.logger.Log('\n{}  Initialize parameters . . .'.format(time.ctime()))
    genos = Libs.genos_file_formats.FillMissings(genos)                         # Fill missing SNPs.

    # Initialize STRUCTURE parameters.
    z, q, p = Libs.structure.InitializeParameters(genos, num_clusters)
    prior_lambda = np.ones((num_clusters, Libs.priors.NUM_ALLELES), dtype=np.float) * 0.1
    prior_alpha = np.ones((num_clusters), dtype=np.float) / num_clusters
    cnt_genos, cnt_origs = Libs.structure.CountAllelesAndOrigins(genos, z, num_clusters)
    smpl_z, smpl_q, smpl_p = Libs.structure.InitializeSamples(genos, num_clusters)

    # Initialize BEAM parameters.
    dataC = []
    dataU = []
    mAlpha = []
    mBeta = []
    graph = []
    m_status = []
    graph_samples = []
    samples = np.zeros(num_loci * 2)
    log_p = 0
    for k in range(num_clusters):
        gn = [ genos[n] for n in range(num_indivs) if smpl_q[n].argmax() == k ]
        lbl = [ labels[n] for n in range(num_indivs) if smpl_q[n].argmax() == k ]
        du = [ gn[n] for n in range(len(gn)) if lbl[n] == 0 ]
        dc = [ gn[n] for n in range(len(gn)) if lbl[n] == 1 ]
        ma, mb = Libs.calc_alpha_beta.InitAlphaBeta(gn, lbl, Libs.priors.ALPHA)
        gh, m_st = Libs.initialize_marker_member.Init(gn, lbl, positions, ma, mb)
        log_p += Libs.cal_full_lp.CalFullLP(dc, du, m_st, gh, ma, mb)

        # Save parameters.
        dataU.append(du);  dataC.append(dc)
        mAlpha.append(ma); mBeta.append(mb)
        graph.append(gh);  m_status.append(m_st)

    # Prepare to run MCMC sampling loop.
    Libs.logger.Log('{}  Start of MCMC . . .'.format(time.ctime()))

    cnt_thining = Libs.confs.THINNING
    old_log_p = log_p
    diff_log_p = 0
    llhood = Libs.structure.CalculateLikelihood(genos, z, q, p)
    is_str_converged = False
    str_cnt_smpls = 0
    beam_cnt_smpls = 0
    for itr in range(1, Libs.confs.BURNIN_ITERS + Libs.confs.MAX_ITERS + 1):    # Start of MCMC
        # Log progress.
        if itr == 1 or (itr % Libs.confs.LOG_ITERS) == 0:
            loop_str = 'BURNIN' if itr < Libs.confs.BURNIN_ITERS else 'MCMC'
            fmt_tuple = (time.ctime(), loop_str, itr, log_p, diff_log_p, beam_cnt_smpls)
            Libs.logger.Log('%s  %s  itr #%d    logP -> %f   diff -> %f    #smpls -> %d' % fmt_tuple)

        # Update STRUCTURE parameters.
        if not is_str_converged:
            # Update P.
            for l in range(num_loci):
                tmp_sm = prior_lambda + cnt_genos[:, l, :]
                for k in range(num_clusters):
                    p[k, l, :] = st.dirichlet.rvs(tmp_sm[k])

            # Update Q and Z.
            is_q_changed = False
            for i in range(num_indivs):
                old_q = q[i, :].argmax()
                q[i, :] = st.dirichlet.rvs(prior_alpha + cnt_origs[i, :])
                new_q = q[i, :].argmax()
                if old_q != new_q:
                    is_q_changed = True
                lg_q = np.log(q[i, :])
                for l in range(num_loci):
                    g = genos[i][l]
                    lg_p = np.log(p[:, l, g])
                    old_z = z[i, l]
                    new_z = z[i, l] = Libs.structure.DrawSampleZ(lg_q + lg_p)
                    if old_z != new_z:
                        cnt_genos[old_z, l, g] -= 1
                        cnt_genos[new_z, l, g] += 1
                        cnt_origs[i, old_z] -= 1
                        cnt_origs[i, new_z] += 1

            # Update parameters relied on Q.
            if is_q_changed:
                dataC = []
                dataU = []
                mAlpha = []
                mBeta = []
                for k in range(num_clusters):
                    gn  = [  genos[n] for n in range(num_indivs) if smpl_q[n].argmax() == k ]
                    lbl = [ labels[n] for n in range(num_indivs) if smpl_q[n].argmax() == k ]
                    du  = [     gn[n] for n in range(len(gn))    if lbl[n] == 0 ]
                    dc  = [     gn[n] for n in range(len(gn))    if lbl[n] == 1 ]
                    ma, mb = Libs.calc_alpha_beta.InitAlphaBeta(gn, lbl, Libs.priors.ALPHA)
                    dataU.append(du);  dataC.append(dc)
                    mAlpha.append(ma); mBeta.append(mb)

            # Save samples if we are in MCMC loop.
            if itr > Libs.confs.BURNIN_ITERS:
                cnt_thining += 1
                if cnt_thining > max(1, Libs.confs.THINNING):
                    cnt_thining = 1
                    str_cnt_smpls += 1
                    smpl_p += p
                    smpl_q += q
                    for i in range(num_indivs):
                        for l in range(num_loci):
                            k = z[i, l]
                            smpl_z[i, l, k] += 1

            # Calculate log likelihood.
            new_llhood = Libs.structure.CalculateLikelihood(genos, z, q, p)
            diff_llhood = abs(new_llhood - llhood)
            llhood = new_llhood

            # Check STRUCTURE convergence.
            if Libs.structure.IsConverged(diff_llhood, itr, str_cnt_smpls):
                fmt_tuple = (time.ctime(), itr, llhood, diff_llhood, str_cnt_smpls)
                Libs.logger.Log('%s  Structure converged    itr #%d    llhood -> %f    diff -> %f #smpls -> %d' % fmt_tuple)
                is_str_converged = True

        # Update association mapping parameters.
        for k in range(num_clusters):
            dc_k = dataC[k]
            du_k = dataU[k]
            g_k = graph[k]
            mAlpha_k = mAlpha[k]
            mBeta_k = mBeta[k]
            m_st_k = m_status[k]
            if 0 in [ len(dc_k), len(du_k), len(mAlpha_k), len(mBeta_k), len(m_st_k) ]:
                continue

            for l in range(num_loci):                                           # Switch membership of an unassociated marker.
                m_l = m_st_k[l]
                dlp, new_m_st = Libs.update_marker_member.UpdateMarkerMember(dc,
                        du, g_k, l, m_l, mAlpha_k, mBeta_k)
                m_st_k[l] = new_m_st
                log_p += dlp

            for i in range(g_k.GetNumCliques()):                                # Switch membership of a marker of a clique.
                c_k = random.choice(range(g_k.GetNumCliques()))
                l = random.choice(g_k.GetMarkers(c_k))
                Libs.switch_marker_member.SwitchMarkerMember(dc, du, positions,
                        g_k, l, m_st_k, mAlpha_k, mBeta_k)

            for i in range(g_k.GetTotalNumMarkers()):                           # Switch membership of two markers of two cliques.
                c_k = random.choice(range(g_k.GetNumCliques()))
                l = random.choice(g_k.GetMarkers(c_k))
                m_l = m_st_k[l]
                dlp, new_m_st = Libs.update_marker_member.UpdateMarkerMember(dc,
                        du, g_k, l, m_l, mAlpha_k, mBeta_k)
                m_st_k[l] = new_m_st
                log_p += dlp

            # Save samples if we are in MCMC loop.
            if itr > Libs.confs.BURNIN_ITERS:
                cnt_thining += 1
                if cnt_thining > max(1, Libs.confs.THINNING):
                    cnt_thining = 1
                    beam_cnt_smpls += 1

                    # Add graph sample.
                    g_smpl = g_k.GetGraphSample()
                    for s in g_smpl:
                        locus_off = num_loci if len(s.markers) > 1 else 0
                        for m in s.markers:
                            samples[m + locus_off] += 1
                    graph_samples.append(g_smpl)

        # Calculate log likelihood.
        diff_log_p = abs(log_p - old_log_p)
        old_log_p = log_p

        # Check BEAM convergence.
        if Libs.BEAM.IsConverged(diff_log_p, itr, beam_cnt_smpls):
            fmt_tuple = (time.ctime(), itr, log_p, diff_log_p, beam_cnt_smpls)
            Libs.logger.Log('%s  Break itr #%d    log_p -> %f    diff -> %f #smpls -> %d' % fmt_tuple)
            break

    Libs.logger.Log('%s  End of MCMC!' % time.ctime())
    Libs.logger.Log('  Num Samples : Str:%d     BEAM:%d' % (str_cnt_smpls, beam_cnt_smpls))

    # Write samples to files.
    Libs.structure.WriteParamsToFile(Libs.logger.LOG_OUT_DIR, genos_path, smpl_z, smpl_q, smpl_p, str_cnt_smpls)    # Write clusters to file.
    Libs.outputs.ClusterResults(graph_samples, positions, 1, is_verbose)                                        # Write graph to file.
    Libs.outputs.DumpPosterior('py-beam-posterior.txt', samples / beam_cnt_smpls, positions)                         # Write posterior to file.

    # TODO: Calculate accuracy of association mapping.
    # Calculate accuracy of clustering.
    if num_clusters < 6:
        clusters = [ smpl_q[i].argmax() for i in range(num_indivs) ]
        clusters_cnt = Libs.calc_accuracy.GetClusterCounts(num_clusters, clusters)
        accs = Libs.calc_accuracy.CalcAccuracy(clusters_cnt)
        mx_acc = int(max(accs) * 100.0)
        Libs.logger.Log('  Acc : %d%%   %s' % (mx_acc, str(np.round(accs, 2))))
    else:
        Libs.logger.Log('  Acc : 0')

