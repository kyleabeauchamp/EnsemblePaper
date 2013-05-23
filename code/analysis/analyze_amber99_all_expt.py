from fitensemble import lvbp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3
import experiment_loader

ff = "amber99"
prior = "maxent"
regularization_strength = 5.0

data_directory = "/%s/%s/" % (ALA3.data_dir, ff)
model_directory = "/%s/%s/models-all-expt-%s/" % (ALA3.data_dir, ff, prior)

predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(data_directory, select_keys=ALA3.all_keys, stride=10)
lvbp_model = lvbp.LVBP.load(model_directory + "/reg-%d-BB0.h5" % regularization_strength)

a = lvbp_model.mcmc.trace("alpha")[:]
plot(a[:,0])

s = lvbp_model.trace_observable(state_ind.T)
