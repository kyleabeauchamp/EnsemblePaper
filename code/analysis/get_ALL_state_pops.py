import numpy as np
import experiment_loader
import ALA3
from fitensemble import belt
import itertools
import sys

bayesian_bootstrap_run = 0
num_threads = 3
rank = int(sys.argv[1])
grid = itertools.product(ALA3.ff_list, ALA3.prior_list)

for k, (ff, prior) in enumerate(grid):
    if k % num_threads == rank:
        print(ff, prior)
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        predictions, measurements, uncertainties = experiment_loader.load(ff)
        phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)
        pymc_filename = ALA3.data_directory + "/models/model_%s_%s_reg-%.1f-BB%d.h5" % (ff, prior, regularization_strength, bayesian_bootstrap_run)
        belt_model = belt.BELT.load(pymc_filename)
        state_pops_trace = belt_model.trace_observable(state_ind.T)
        state_pops = state_pops_trace.mean(0)
        state_uncertainties = state_pops_trace.std(0)
        out_directory = ALA3.data_directory + "state_populations/"
        np.savez_compressed(out_directory + "state_populations_%s_%s_reg_%.1f_BB%d.npz" % (ff, prior, ALA3.regularization_strength_dict[prior][ff], bayesian_bootstrap_run), state_pops_trace)
