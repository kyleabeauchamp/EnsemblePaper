import pandas as pd
import numpy as np
from fitensemble import bayesian_weighting
import experiment_loader
import sys
import ALA3

prior = "BW"
ff = "oplsaa"

out_dir = ALA3.data_directory + "/BW_models/"

predictions_framewise, measurements, uncertainties = experiment_loader.load(ff)
phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)

num_states = ass_raw.max() + 1

prior_pops = np.ones(num_states)
raw_pops = np.bincount(ass_raw).astype('float')
raw_pops /= raw_pops.sum()

predictions = pd.DataFrame(bayesian_weighting.framewise_to_statewise(predictions_framewise, ass_raw), columns=predictions_framewise.columns)
model = bayesian_weighting.BayesianWeighting(predictions.values, measurements.values, uncertainties.values, ass_raw, prior_pops=prior_pops)
model.sample(2000000, thin=ALA3.thin, burn=ALA3.burn)

population_trace_filename = out_dir + "/%s-populations.npz" % ff
pi = model.mcmc.trace("matrix_populations")[:,0]
np.savez_compressed(population_trace_filename, pi)



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
F.write("all,BW4,%s,%s,%f \n" % (ff, prior, chi2_all))
F.write("train,BW4,%s,%s,%f \n" % (ff, prior, chi2_train))
F.write("test,BW4,%s,%s,%f \n" % (ff, prior, chi2_test))
F.flush()

