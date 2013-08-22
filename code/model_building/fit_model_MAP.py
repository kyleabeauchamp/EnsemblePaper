import pandas as pd
import pymc
import numpy as np
from fitensemble import belt, ensemble_fitter
import experiment_loader
import sys
import ALA3
belt.ne.set_num_threads(3)

regularization_strength = 1E-6
prior = "maxent"
pops = {}

for ff in ALA3.ff_list:
    print(ff)

    phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)
    predictions, measurements, uncertainties = experiment_loader.load(ff)

    num_frames, num_measurements = predictions.shape
    bootstrap_index_list = np.array_split(np.arange(num_frames), ALA3.num_blocks)

    prior_pops = None

    model = belt.MaxEntBELT(predictions.values, measurements.values, uncertainties.values, regularization_strength, prior_pops=prior_pops)

    model.sample(10)

    MAP = pymc.MAP(model)
    MAP.fit()

    alpha = MAP.alpha.value

    p = belt.get_populations_from_alpha(alpha, predictions.values, model.prior_pops)
    pops[ff] = p[ass_raw == 0].sum(), p[ass_raw == 1].sum(), p[ass_raw == 2].sum()

pops = pd.DataFrame(pops)
