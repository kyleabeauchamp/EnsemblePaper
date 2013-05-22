import pandas as pd
import numpy as np
import experiment_loader
import ALA3
from fitensemble import lvbp

ff = "amber96"

directory = ALA3.data_dir + "/%s/" % ff
predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory)
lvbp_model = lvbp.LVBP.load("test.h5")



x = pymc.Normal("x", mu=0, tau=1)
model = pymc.MCMC([x], db="hdf5")
