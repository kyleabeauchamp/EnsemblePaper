import pandas as pd
import itertools
import numpy as np
import experiment_loader
import ALA3
from fitensemble import belt
import sys

bayesian_bootstrap_run = 1
num_threads = 3
rank = int(sys.argv[1])
grid = itertools.product(ALA3.ff_list, ALA3.prior_list)
grid = [("amber99sbnmr-ildn", "MVN")]

for k, (ff, prior) in enumerate(grid):
    if k % num_threads == rank:
        print(ff, prior, bayesian_bootstrap_run)
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        predictions, measurements, uncertainties = experiment_loader.load(ff, keys=None)
        phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)
        pymc_filename = ALA3.data_directory + "/models/model_%s_%s_reg-%.1f-BB%d.h5" % (ff, prior, regularization_strength, bayesian_bootstrap_run)
        belt_model = belt.BELT.load(pymc_filename)

        mu_mcmc = belt_model.trace_observable(predictions)
        mu_mcmc = pd.DataFrame(mu_mcmc, columns=measurements.index)
        
        out_filename = "mcmc_traces/mu_%s_%s_reg-%.1f-BB%d.h5" % (ff, prior, regularization_strength, bayesian_bootstrap_run)
        mu_mcmc.to_hdf(out_filename, "data")
