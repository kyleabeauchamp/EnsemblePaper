from fitensemble import belt
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3
import experiment_loader

""" NOTE: I should be using the MCMC samples, not the average alpha."""

rms_raw = np.zeros((len(ALA3.prior_list), len(ALA3.ff_list)))
rms_belt = np.zeros((len(ALA3.prior_list), len(ALA3.ff_list)))
for i, prior in enumerate(ALA3.prior_list):
    for j, ff in enumerate(ALA3.ff_list):
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        data_directory = "/%s/%s/" % (ALA3.data_dir, ff)
        model_directory = "/%s/%s/models-%s/" % (ALA3.data_dir, ff, prior)
        p = np.loadtxt(model_directory + "reg-%d-frame-populations.dat" % regularization_strength)
        predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(data_directory, select_keys=ALA3.test_keys)
        belt_model = belt.BELT.load(model_directory + "/reg-%d-BB0.h5" % regularization_strength)
        mu_belt = belt_model.trace_observable(predictions)
        data = pd.DataFrame(mu_belt, columns=predictions.columns)
        data.to_hdf(model_directory + "test_set_mcmc.h5", 'data')
        print(prior, ff)
        #print((predictions.T.dot(p) - measurements) / uncertainties)
        #rms_raw[i,j] = np.mean(((predictions.mean(0) - measurements) / uncertainties) ** 2.0) ** 0.5
        #rms_belt[i,j] = np.mean(((predictions.T.dot(p) - measurements) / uncertainties) ** 2.0) ** 0.5  # Don't use average populations, use MCMC
        #rms_belt[i,j] = np.mean(((mu_belt - measurements) / uncertainties) ** 2.0).mean() ** 0.5
        #print(rms_belt[i,j])

rms_raw = pd.DataFrame(rms_raw, columns=ALA3.ff_list, index=ALA3.prior_list)
rms_belt = pd.DataFrame(rms_belt, columns=ALA3.ff_list, index=ALA3.prior_list)
