import numpy as np
from fitensemble import lvbp
import experiment_loader
import sys
import ALA3

regularization_strength = 5.0
prior = "maxent"
ff = "amber99"
bayesian_bootstrap_run = 0
num_samples = 100000000
stride = 10
thin = 1000

directory = "%s/%s" % (ALA3.data_dir , ff)
out_dir = directory + "/models-all-expt-%s/" % prior
pymc_filename = out_dir + "/reg-%d-BB%d.h5" % (regularization_strength, bayesian_bootstrap_run)

predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, stride=stride, select_keys=ALA3.all_keys)
print(predictions)
num_frames, num_measurements = predictions.shape

model = lvbp.MaxEnt_LVBP(predictions.values, measurements.values, uncertainties.values, regularization_strength)
model.sample(num_samples, thin=thin, burn=ALA3.burn, filename=pymc_filename)

p = model.accumulate_populations()
np.savetxt(out_dir+"/reg-%d-frame-populations.dat" % (regularization_strength), p)
