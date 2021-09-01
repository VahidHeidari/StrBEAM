import numpy as np



DEF_CHI_2_PRIOR = 1e-7



def GetChiSquare(counts, prior=DEF_CHI_2_PRIOR):
    rows = len(counts)
    cols = len(counts[0])

    t = np.sum(counts) + (rows * cols * prior)                                  # Total
    rows_sum = np.sum(counts, 1) + (cols * prior)

    chi = 0                                                                     # Chi square calculate
    cols_sum = np.sum(counts, 0) + (rows * prior)
    for r in range(rows):
        for c in range(cols):
            obs = counts[r][c] + prior
            m = float(cols_sum[c] * rows_sum[r]) / t
            chi += ((obs - m) ** 2) / m
    return chi



if __name__ == '__main__':
    print('This is a module!')

