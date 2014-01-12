import numpy as np
from fitensemble import belt, ensemble_fitter
import experiment_loader
import sys
import ALA3
belt.ne.set_num_threads(2)

num_samples = 10000
ff = "amber96"
prior = "maxent"
regularization_strength = 5.0


predictions, measurements, uncertainties = experiment_loader.load(ff, keys=[(u'JC', 2, u'J3_HN_Cprime')])

num_frames, num_measurements = predictions.shape
bootstrap_index_list = np.array_split(np.arange(num_frames), ALA3.num_blocks)
prior_pops = None

model = belt.MaxEntBELT(predictions.values, measurements.values, uncertainties.values, regularization_strength, prior_pops=prior_pops)
%time model.sample(num_samples, thin=ALA3.thin, burn=ALA3.burn)


num_frames = 50000
num_measurements = 76
predictions = np.random.normal(size=(num_frames, num_measurements))
measurements = np.zeros(num_measurements)
uncertainties = np.ones(num_measurements)

model = belt.MaxEntBELT(predictions, measurements, uncertainties, regularization_strength, prior_pops=prior_pops)
%time model.sample(num_samples, thin=ALA3.thin, burn=ALA3.burn)
