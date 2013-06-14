import numpy as np
from fitensemble import lvbp, ensemble_fitter
import experiment_loader
import ALA3
lvbp.ne.set_num_threads(2)

ff = "amber99"
regularization_strength = 0.1

predictions, measurements, uncertainties = experiment_loader.load(ff, keys=[("JC",2,"J3_HN_HA"), ("JC",2,"J3_HN_Cprime")])
num_frames, num_measurements = predictions.shape
phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, 1)

model = lvbp.MaxEnt_LVBP(predictions.values, measurements.values, uncertainties.values, regularization_strength)

model.sample(10000000, thin=50, burn=1000)

si = model.trace_observable(state_ind.T)
np.savez_compressed("./amber99_two_expt.npz", si)
