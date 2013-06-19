import numpy as np
import experiment_loader
import ALA3
from fitensemble import belt
import itertools
import sys

bayesian_bootstrap_run = 0
num_threads = 2
rank = int(sys.argv[1])
grid = itertools.product(ALA3.ff_list, ALA3.prior_list)

for k, (ff, prior) in enumerate(grid):
    if k % num_threads == rank:
        print(ff, prior)
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        pymc_filename = ALA3.data_directory + "/models/model_%s_%s_reg-%.1f-BB%d.h5" % (ff, prior, regularization_strength, bayesian_bootstrap_run)
        belt_model = belt.BELT.load(pymc_filename)
        frame_pops = belt_model.accumulate_populations()
        np.savetxt(ALA3.data_directory + "/frame_populations/model_%s_%s_reg-%.1f-BB%d.h5" % (ff, prior, regularization_strength, bayesian_bootstrap_run), frame_pops)
