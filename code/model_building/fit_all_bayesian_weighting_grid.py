import schwalbe_couplings
import pandas as pd
import numpy as np
from fitensemble import bayesian_weighting
import experiment_loader
import sys
import ALA3

effective_counts = 1000.
num_bins = 12
num_states = num_bins ** 2
prior = "BW%d" % num_bins
ALA3.ff_list = ["oplsaa"]
for ff in ALA3.ff_list:
    directory = "%s/%s" % (ALA3.data_dir , ff)
    out_dir = directory + "/models-%s/" % prior
    pymc_filename = out_dir + "/model.h5"

    predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, stride=ALA3.stride)
    num_frames, num_measurements = predictions.shape

    assignments = schwalbe_couplings.assign_grid(phi, psi, num_bins)[2]
    prior_state_pops = np.bincount(assignments).astype('float')
    prior_state_pops /= prior_state_pops.sum()
    prior_state_pops *= effective_counts
     
    model = bayesian_weighting.BayesianWeighting(predictions.values, measurements.values, uncertainties.values, assignments, prior_state_pops=prior_state_pops)
    model.sample(ALA3.bw_num_samples, thin=ALA3.thin, burn=ALA3.burn, filename=pymc_filename)
    p = model.accumulate_populations()

    pi = model.mcmc.trace("matrix_populations[0]")[:]
    mu = model.mcmc.trace("mu")[:]
    rms = (((mu - measurements.values) / uncertainties.values)**2).mean(0).mean()**0.5

    predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, select_keys=ALA3.test_keys)
    predictions_statewise = np.array([predictions.values[assignments == i].mean(0) for i in np.arange(model.num_states)])
    mu_test = model.trace_observable(predictions_statewise)
    rms_test = (((mu_test - measurements.values) / uncertainties.values)**2).mean(0).mean()**0.5
    data = pd.DataFrame(mu_test, columns=predictions.columns)
    data.to_hdf(out_dir + "test_set_mcmc.h5", "data")
