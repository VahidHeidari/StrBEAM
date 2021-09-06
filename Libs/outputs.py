import copy
import os
import unittest

import numpy as np

import logger
import disease_graph



def DumpPosterior(out_path, samples, positions):
    num_loci = len(positions)
    with open(os.path.join(logger.LOG_OUT_DIR, out_path), 'w') as f:            # Write posterior to file.
        for i in range(num_loci):
            f.write('{}\t{} {}\t{} + {} = {}\n'.format(i, positions[i][0],
                positions[i][1], samples[i], samples[i + num_loci],
                samples[i] + samples[i + num_loci]))
    # TODO: Plot posterior.


def DumpDotFile(out_path, nodes, intn, rep, sm, positions):
    with open(os.path.join(logger.LOG_OUT_DIR, out_path), 'w') as f:
        f.write('graph {\n')
        f.write('node [shape = circle];' + ' '.join(['s%d' % i for i in range(len(nodes))]) + ';\n')
        for i in range(len(nodes)):
            chrom = positions[rep[i]][0]
            pos = positions[rep[i]][1]
            fmt_tuple = (i, rep[i], sm[nodes[i]], chrom, pos)
            fmt_str = '\ts%d [label = "snp%d (%4.3f)\\n%d:%d"]\n' % fmt_tuple
            f.write(fmt_str)
        for i in range(len(intn)):
            for j in range(i, len(intn)):
                if intn[i][j] > 0.05:
                    f.write('\ts%d -- s%d [label = "%4.3f"]\n' % (i, j, intn[i][j]))
        f.write('}')


def ClusterResults(graph_samples, positions, interval=100000, is_verbose=True):
    num_loci = len(positions)
    num_graph_samples = len(graph_samples)
    if num_graph_samples == 0:
        logger.Log('Warning: Number of samples is zero!')
        return

    prb = np.zeros(num_loci)
    for rec in graph_samples:
        for clq in rec:
            for m in clq.markers:
                prb[m] += 1
    prb /= num_graph_samples

    sm = np.zeros(num_loci)
    lb = np.zeros(num_loci, dtype=np.int)
    rb = np.zeros(num_loci, dtype=np.int)
    st = 0
    for ed in range(num_loci):
        chrom = positions[ed][0]
        pos = positions[ed][1]
        if chrom != positions[0][0] or pos != positions[0][1] + interval // 2:
            break
        sm[0] += prb[ed]
    lb[0] = st
    rb[0] = ed
    for i in range(1, num_loci):
        sm[i] = sm[i - 1]
        while st < i:
            chrom = positions[st][0]
            pos = positions[st][1]
            if chrom != positions[i][0] or pos < positions[i][1] - interval // 2:
                sm[i] -= prb[st]
            else:
                break
            st += 1
        while ed < num_loci:
            chrom = positions[ed][0]
            pos = positions[ed][1]
            if chrom != positions[i][0] or pos > positions[i][1] + interval // 2:
                break
            sm[i] += prb[ed]
            ed += 1
        lb[i] = st
        rb[i] = ed

    lst = []
    for i in range(num_loci):
        if i > 0 and lb[i] <= i - 1 and sm[i] < sm[i - 1]:
            continue
        if i < num_loci - 1 and rb[i] > i + 1 and sm[i] < sm[i + 1]:
            continue
        if sm[i] > 0.05:
            lst.append(i)
    i = len(lst) - 2
    while i >= 0:
        flg = False
        for j in range(lb[lb[lst[i]]], lst[i]):
            if sm[j] > sm[lst[i]]:
                flg = True
                break
        for j in range(lst[i] + 1, rb[rb[lst[i]]] - 1):
            if sm[j] > sm[lst[i]]:
                flg = True
                break
        if flg:
            del lst[i]
        i -= 1

    rep = []
    for i in range(len(lst)):
        mi = lb[lst[i]]
        for j in range(lb[lst[i]] + 1, rb[lst[i]]):
            if prb[mi] < p[j]:
                mi = j
        rep.append(mi)

    intn = []
    if len(lst) > 0:
        intn = np.zeros((len(lst), len(lst)))
        for i in range(num_graph_samples):
            num_cliques = len(graph_samples[i])
            dmap = [ [] for _ in range(num_cliques) ]
            for j in range(num_cliques):
                tmpd = copy.deepcopy(graph_samples[i][j].markers)
                for x in range(len(tmpd)):
                    l = tmpd[x]
                    tmpd[x] = -1
                    for y in range(len(lst)):
                        if lb[lst[y]] <= l and l < rb[lst[y]]:
                            tmpd[x] = y
                            break
                x = len(tmpd) - 1
                while x >= 0:
                    if tmpd[x] < 0:
                        del tmpd[x]
                    x -= 1
                dmap[j] = tmpd
            for j in range(num_cliques):
                for x in range(len(dmap[j]) - 1):
                    a = dmap[j][x]
                    for y in range(x + 1, len(dmap[j])):
                        b = dmap[j][y]
                        intn[min(a, b)][max(a, b)] += 1
                for k in range(len(graph_samples[i][j].connections)):
                    if graph_samples[i][j].connections[k] > j:
                        l = graph_samples[i][j].connections[k]
                        for x in range(len(dmap[j])):
                            a = dmap[j][x]
                            for y in range(len(dmap[l])):
                                b = dmap[l][y]
                                intn[min(a, b)][max(a, b)] += 1
        intn /= num_graph_samples

    if is_verbose:
        with open(os.path.join(logger.LOG_OUT_DIR, 'py-sum.txt'), 'w') as f:
            f.write('\n'.join(['chr:{}\tpos:{}\tp:{}\tsum:{}'.format(positions[l][0],
                positions[l][1], prb[l], sm[l]) for l in range(num_loci)]))
        with open(os.path.join(logger.LOG_OUT_DIR, 'py-site.txt'), 'w') as f:
            f.write('\n'.join(['list:{}\trep:{}\tsum[list]:{}'.format(lst[i],
                rep[i], sm[lst[i]]) for i in range(len(lst))]))
        with open(os.path.join(logger.LOG_OUT_DIR, 'py-intn.txt'), 'w') as f:
            for i in range(len(intn)):
                f.write(' '.join(['{:4.3f}'.format(intn[i][j]) for j in range(len(intn[i]))]) + '\n')

    DumpDotFile('py-beam3-g.dot', lst, intn, rep, sm, positions)



class ClusterResultsTestCase(unittest.TestCase):
    def test_ClusterResults(self):
        is_verbose = True
        positions = [           # Make positions. position[n][0]:chr, position[n][1]:pos
            (1, 1),		# 0
            (1, 2),		# 1
            (1, 3),		# 2
            (1, 6),		# 3
            (1, 7),		# 4
            (1, 8),		# 5
        ]

	#
	# Record 0:
	#        .---.       .---.
	#       / n0  \_____/  n1 \
	#       \ {1} /     \ {4} /
	#        '---'       '---'
	#
        graph_samples = [ [ disease_graph.GraphSample([1], [1]), disease_graph.GraphSample([4], [1]) ] ]
        ClusterResults(graph_samples, positions, 1, is_verbose)                 # Write graph to file.


if __name__ == '__main__':
    unittest.main()

