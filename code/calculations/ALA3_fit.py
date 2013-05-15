import numpy as np
from fit_ensemble import lvbp, ensemble_fitter
import experiment_loader
import sys
import ALA3

ff = sys.argv[1]
regularization_strength = float(sys.argv[2])
bayesian_bootstrap_run = int(sys.argv[3])

directory = "/home/kyleb/dat/lvbp/%s/" % ff
out_dir = directory + "/models-%s/" % ALA3.model
pymc_filename = out_dir + "/reg-%d-BB%d.h5" % (regularization_strength, bayesian_bootstrap_run)

keys, measurements, predictions, uncertainties, phi, psi = experiment_loader.load(directory, stride=ALA3.stride)

num_frames, num_measurements = predictions.shape
bootstrap_index_list = np.array_split(np.arange(len(predictions)), ALA3.kfold)

if bayesian_bootstrap_run == 0:
    prior_pops = None
else:
    prior_pops = ensemble_fitter.sample_prior_pops(num_frames, bootstrap_index_list)

S = lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, regularization_strength, prior_pops=prior_pops)

S.sample(ALA3.num_samples, thin=ALA3.thin, burn=ALA3.burn, filename=pymc_filename)
p = S.accumulate_populations()

np.savetxt(out_dir+"/reg-%d-frame-populations.dat" % (regularization_strength), p)