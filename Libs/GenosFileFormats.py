import os



def ReadBEAM(in_path):
    lines = [ l.strip() for l in open(in_path, 'r') ]                           # Read file lines.
    lbls = [ int(l) for l in lines[0].split()[3:] ]                             # Read labels.
    snps = [ [ int(l) for l in i.split()[3:] ] for i in lines[1:] ]             # Read SNPs.

    num_loci = len(lbls)                                                        # Transpose snps.
    num_indivs = len(snps)
    genos = [ [ snps[l][i] for l in range(num_loci) ] for i in range(num_indivs) ]
    return genos, lbls


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

        f.write('rs{} Chr1 {} '.format(l + 1, l))                               # Write SNPs.
        f.write(' '.join([ str(genos[i][0]) for i in range(num_indivs) ]))
        for l in range(1, num_loci):
            f.write('\nrs{} Chr1 {} '.format(l + 1, l))
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
        with open(out_path + 'labels.txt', 'w') as f:
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



if __name__ == '__main__':
    print('This is a module!')

