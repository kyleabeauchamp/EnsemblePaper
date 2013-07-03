import os
import numpy as np
from fitensemble import belt, ensemble_fitter
import experiment_loader
import ALA3
belt.ne.set_num_threads(2)

regularization_strength = 0.1
predictions, measurements, uncertainties = experiment_loader.load("amber99", keys=None)
all_keys = measurements.index

for ff in ALA3.ff_list[::-1]:
    for i, key0 in enumerate(all_keys):
        for j, key1 in enumerate(all_keys):
            if i > j:
                print(ff, i, j, key0, key1)
                keys = [key0, key1]
                predictions, measurements, uncertainties = experiment_loader.load(ff, keys=keys)
                m0 = str(measurements.index[0])
                m1 = str(measurements.index[1])
                filename = "./pair_experiments/%s-%s-%s" % (ff, m0, m1)
                if not os.path.exists(filename):
                    num_frames, num_measurements = predictions.shape
                    phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, 1)
                    model = belt.MVN_BELT(predictions.values, measurements.values, uncertainties.values, regularization_strength)
                    model.sample(150000, thin=5, burn=1000)
                    si = model.trace_observable(state_ind.T)
                    np.savez_compressed(filename, si)
