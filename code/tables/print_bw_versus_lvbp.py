from fitensemble import lvbp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3
import experiment_loader

""" NOTE: I should be using the MCMC samples, not the average alpha."""

prior_list = ["maxent", "MVN", "BW", "BW12"]
rms_raw = np.zeros((len(prior_list), len(ALA3.ff_list)))
rms_lvbp = np.zeros((len(prior_list) + 1, len(ALA3.ff_list)))
for i, prior in enumerate(prior_list):
    for j, ff in enumerate(ALA3.ff_list):
        print(prior, ff)
        data_directory = "/%s/%s/" % (ALA3.data_dir, ff)
        model_directory = "/%s/%s/models-%s/" % (ALA3.data_dir, ff, prior)
        predictions, measurements, uncertainties = experiment_loader.load(data_directory, keys=ALA3.test_keys)
        mu_lvbp = pd.io.pytables.HDFStore(model_directory + "test_set_mcmc.h5",'r')["data"]
        #print((predictions.T.dot(p) - measurements) / uncertainties)
        rms_raw[i,j] = (((predictions.mean(0) - measurements) / uncertainties) ** 2.0).mean() ** 0.5
        rms_lvbp[i,j] = (((mu_lvbp - measurements) / uncertainties) ** 2.0).mean().mean() ** 0.5

columns = []
columns.extend(prior_list)
columns.append("MD")
rms_lvbp[-1] = rms_raw[0]
#rms_raw = pd.DataFrame(rms_raw, columns=ALA3.ff_list, index=prior_list)
rms_lvbp = pd.DataFrame(rms_lvbp, columns=ALA3.ff_list, index=columns)
