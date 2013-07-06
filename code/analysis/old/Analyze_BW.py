import numpy as np
from fitensemble import bayesian_weighting
import experiment_loader
import sys
import ALA3

prior = "BW"
ff = "amber96"
effective_counts = 1.

out_dir = ALA3.data_directory + "/BW_models/"
pymc_filename = out_dir + "/model_BW_%s.h5" % (ff)

predictions, measurements, uncertainties = experiment_loader.load(ff)
phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)
num_frames, num_measurements = predictions.shape

num_states = ass_raw.max() + 1
prior_state_pops = np.bincount(ass_raw).astype('float')
prior_state_pops /= prior_state_pops.sum()
prior_state_pops *= effective_counts

prior_state_pops = np.ones(num_states)

model = bayesian_weighting.BayesianWeighting.load(pymc_filename)

mu = model.mcmc.trace("mu")[:]
chi2 = (((mu - measurements.values) / uncertainties.values)**2).mean(0).mean()

predictions_test, measurements_test, uncertainties_test = experiment_loader.load(ff, keys=ALA3.test_keys)
predictions_test_statewise = np.array([predictions_test.values[ass_raw == i].mean(0) for i in np.arange(model.num_states)])

mu_test = model.trace_observable(predictions_test_statewise)
chi2_test = (((mu_test - measurements_test.values) / uncertainties_test.values)**2).mean(0).mean()
chi2_test_raw = (((predictions_test.mean(0) - measurements_test.values) / uncertainties_test.values)**2).mean(0)
