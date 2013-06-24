from fitensemble import lvbp
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3
import experiment_loader

rms_raw = np.zeros((len(ALA3.prior_list), len(ALA3.ff_list)))
rms_lvbp = np.zeros((len(ALA3.prior_list), len(ALA3.ff_list)))
for i, prior in enumerate(ALA3.prior_list):
    for j, ff in enumerate(ALA3.ff_list):
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        data_directory = "/%s/%s/" % (ALA3.data_dir, ff)
        model_directory = "/%s/%s/models-%s/" % (ALA3.data_dir, ff, prior)
        p = np.loadtxt(model_directory + "reg-%d-frame-populations.dat" % regularization_strength)
        predictions, measurements, uncertainties = experiment_loader.load(data_directory, keys=ALA3.test_keys)
        lvbp_model = lvbp.LVBP.load(model_directory + "/reg-%d-BB0.h5" % regularization_strength)
        mu_lvbp = lvbp_model.trace_observable(predictions)
        data = pd.DataFrame(mu_lvbp, columns=predictions.columns)
        print(data)
        data.to_hdf(model_directory + "test_set_mcmc.h5", 'data', mode="w")
        print(prior, ff)

rms_raw = pd.DataFrame(rms_raw, columns=ALA3.ff_list, index=ALA3.prior_list)
rms_lvbp = pd.DataFrame(rms_lvbp, columns=ALA3.ff_list, index=ALA3.prior_list)
