import numpy as np
from fitensemble import belt, ensemble_fitter
import experiment_loader
import ALA3
belt.ne.set_num_threads(2)

ff = "oplsaa"
regularization_strength = 0.1

predictions, measurements, uncertainties = experiment_loader.load(ff, keys=[("JC",2,"J3_HN_CB"), ("CS",2,"HA")])
num_frames, num_measurements = predictions.shape
phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, 1)

model = belt.MaxEnt_BELT(predictions.values, measurements.values, uncertainties.values, regularization_strength)

model.sample(2500000, thin=20, burn=1000)

si = model.trace_observable(state_ind.T)
np.savez_compressed("./%s_two_expt.npz" % ff, si)
