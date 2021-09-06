import itertools



def GetClusterCounts(num_clusters, clusters):
    clstr_count = [[0 for k in range(num_clusters)] for i in range(num_clusters)]
    num_indivs_in_cluster = len(clusters) // num_clusters
    for k in range(num_clusters):
        for i in range(num_indivs_in_cluster):
            indiv_idx = k * num_indivs_in_cluster + i
            clstr_idx = clusters[indiv_idx]
            clstr_count[k][clstr_idx] += 1
    return clstr_count


def GetClusterCounts(num_clusters, clusters):
    clstr_count = [[0 for k in range(num_clusters)] for i in range(num_clusters)]
    num_indivs_in_cluster = len(clusters) // num_clusters
    for k in range(num_clusters):
        for i in range(num_indivs_in_cluster):
            indiv_idx = k * num_indivs_in_cluster + i
            clstr_idx = clusters[indiv_idx]
            clstr_count[k][clstr_idx] += 1
    return clstr_count


def GetAcc(prm, cnts):
    tot = 0
    for rec in cnts:
        for c in rec:
            tot += c

    num_clstr = len(cnts[0])
    sm = 0.0
    for k in range(num_clstr):
        idx = prm[k]
        sm += cnts[k][idx]
    acc = sm / tot
    return acc


def CalcAccuracy(clusters_count):
    num_clstr = len(clusters_count[0])
    items = [j for j in range(num_clstr)]
    accs = [GetAcc(prm, clusters_count) for prm in itertools.permutations(items)]
    return accs



if __name__ == '__main__':
    print('This is a module!')

