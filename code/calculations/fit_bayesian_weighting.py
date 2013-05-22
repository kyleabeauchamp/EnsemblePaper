import numpy as np
from fitensemble import bayesian_weighting
import experiment_loader
import sys
import ALA3

prior = "BW"
ff = "oplsaa"
effective_counts = 100

directory = "%s/%s" % (ALA3.data_dir , ff)
out_dir = directory + "/models-%s/" % prior
pymc_filename = out_dir + "/model.h5"

predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, stride=ALA3.stride)
num_frames, num_measurements = predictions.shape

prior_state_pops = np.bincount(ass_raw).astype('float')
prior_state_pops /= prior_state_pops.sum()
prior_state_pops *= effective_counts

model = bayesian_weighting.BayesianWeighting(predictions.values, measurements.values, uncertainties.values, ass_raw, prior_state_pops=prior_state_pops)
#model.sample(ALA3.num_samples, thin=ALA3.thin, burn=ALA3.burn, filename=pymc_filename)
model.sample(5000000, thin=ALA3.thin, burn=ALA3.burn)
p = model.accumulate_populations()

pi = model.mcmc.trace("matrix_populations[0]")[:]
mu = model.mcmc.trace("mu")[:]
rms = (((mu - measurements.values) / uncertainties.values)**2).mean(0).mean()**0.5

predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, select_keys=ALA3.test_keys)
predictions_statewise = np.array([predictions.values[ass_raw == i].mean(0) for i in np.arange(model.num_states)])

mu_test = model.trace_observable(predictions_statewise)
rms_test = (((mu_test - measurements.values) / uncertainties.values)**2).mean(0).mean()**0.5


lvbp_model = lvbp.LVBP.load("./oplsaa/models-maxent/reg-7-BB0.h5")
mu_lvbp = lvbp_model.trace_observable(predictions)
rms_lvbp = (((mu_lvbp - measurements.values) / uncertainties.values)**2).mean(0).mean()**0.5
