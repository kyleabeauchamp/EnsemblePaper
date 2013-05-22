import pandas as pd
import numpy as np
from fitensemble import bayesian_weighting
import experiment_loader
import sys
import ALA3

prior = "BW"
for ff in ALA3.ff_list:
    directory = "%s/%s" % (ALA3.data_dir , ff)
    out_dir = directory + "/models-%s/" % prior
    pymc_filename = out_dir + "/model.h5"

    predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, stride=ALA3.stride)
    num_frames, num_measurements = predictions.shape

    model = bayesian_weighting.BayesianWeighting(predictions.values, measurements.values, uncertainties.values, ass_raw)
    model.sample(ALA3.bw_num_samples, thin=ALA3.thin, burn=ALA3.burn, filename=pymc_filename)
    p = model.accumulate_populations()

    pi = model.mcmc.trace("matrix_populations[0]")[:]
    mu = model.mcmc.trace("mu")[:]
    rms = (((mu - measurements.values) / uncertainties.values)**2).mean(0).mean()**0.5

    predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, select_keys=ALA3.test_keys)
    predictions_statewise = np.array([predictions.values[ass_raw == i].mean(0) for i in np.arange(model.num_states)])
    mu_test = model.trace_observable(predictions_statewise)
    rms_test = (((mu_test - measurements.values) / uncertainties.values)**2).mean(0).mean()**0.5
    data = pd.DataFrame(mu_test, columns=predictions.columns)
    data.to_hdf(out_dir + "test_set_mcmc.h5", "data")
