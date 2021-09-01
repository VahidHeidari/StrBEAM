
# Change below options if needed.
MAX_ITERS = 1000
THINNING = 3
MIN_SAMPLE_SIZE = 100

# These options are not usually needed to be changed.
CONV_EPSILONE = 5e-1

NUM_ALLELES = 3
ALLELE_PRIOR = 1.0 / NUM_ALLELES

LOG_ITERS = max(1, MAX_ITERS // 100)
BURNIN_ITERS = max(5, MAX_ITERS // 3)

