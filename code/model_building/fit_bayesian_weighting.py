import pandas as pd
import numpy as np
from fitensemble import bayesian_weighting
import experiment_loader
import sys
import ALA3

prior = "BW"
ff = "amber99sbnmr-ildn"

out_dir = ALA3.data_directory + "/BW_models/"

predictions_framewise, measurements, uncertainties = experiment_loader.load(ff)
phi, psi, ass_raw0, state_ind0 = experiment_loader.load_rama(ff, ALA3.stride)

state_ind1 = state_ind0[0:3].copy()
ass_raw = ass_raw0.copy()
state_ind1[2][ass_raw == 3] = 2
ass_raw[ass_raw == 3] = 2

#state_ind = state_ind1[0:2]
#state_ind[1][ass_raw == 2] = 1
#ass_raw[ass_raw == 2] = 1

num_states = ass_raw.max() + 1

prior_pops = np.ones(num_states)
raw_pops = np.bincount(ass_raw).astype('float')
raw_pops /= raw_pops.sum()

predictions = pd.DataFrame(bayesian_weighting.framewise_to_statewise(predictions_framewise, ass_raw), columns=predictions_framewise.columns)
model = bayesian_weighting.BayesianWeighting(predictions.values, measurements.values, uncertainties.values, ass_raw, prior_pops=prior_pops)
model.sample(2000000, thin=ALA3.thin, burn=ALA3.burn)

population_trace_filename = out_dir + "/%s-BW3-populations.npz" % ff
state_ind_statewise = np.zeros((3, num_states))
state_ind_statewise[0,0] = 1.
state_ind_statewise[1,1] = 1.
state_ind_statewise[2,2] = 1.
pi = model.trace_observable(state_ind_statewise.T)
#pi = model.mcmc.trace("matrix_populations")[:,0]
np.savez_compressed(population_trace_filename, pi)
print(ff)
pi.mean(0)


mu = model.mcmc.trace("mu")[:]
chi2_train = (((mu - measurements.values) / uncertainties.values) ** 2).mean(0).mean()

predictions_framewise_test, measurements_test, uncertainties_test = experiment_loader.load(ff, keys=ALA3.test_keys)
predictions_test = pd.DataFrame(bayesian_weighting.framewise_to_statewise(predictions_framewise_test, ass_raw), columns=predictions_framewise_test.columns)

mu_test = model.trace_observable(predictions_test)
chi2_test = (((mu_test - measurements_test.values) / uncertainties_test.values) ** 2).mean(0).mean()
chi2_test_raw = (((predictions_test.T.dot(raw_pops) - measurements_test.values) / uncertainties_test.values) ** 2).mean(0)


predictions_framewise_all, measurements_all, uncertainties_all = experiment_loader.load(ff, keys=None)
predictions_all = pd.DataFrame(bayesian_weighting.framewise_to_statewise(predictions_framewise_all, ass_raw), columns=predictions_framewise_all.columns)

mu_all = model.trace_observable(predictions_all)
chi2_all = (((mu_all - measurements_all.values) / uncertainties_all.values) ** 2).mean(0).mean()
chi2_all_raw = (((predictions_all.T.dot(raw_pops) - measurements_all.values) / uncertainties_all.values) ** 2).mean(0)


F = open(ALA3.chi2_filename, 'a')
F.write("all,BW%d,%s,%s,%f \n" % (num_states, ff, prior, chi2_all))
F.write("train,BW%d,%s,%s,%f \n" % (num_states, ff, prior, chi2_train))
F.write("test,BW%d,%s,%s,%f \n" % (num_states, ff, prior, chi2_test))
F.flush()

