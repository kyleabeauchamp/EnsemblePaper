import pandas as pd
import numpy as np
import experiment_loader
import ALA3
from fitensemble import belt
import itertools

num_BB = 2

prior = "maxent"

data = {}
chi2 = {}
for ff in ALA3.ff_list:
    print(ff, prior)
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]
    predictions, measurements, uncertainties = experiment_loader.load(ff, keys=None)
    z = (predictions.mean() - measurements) / uncertainties
    chi2_all = (z ** 2).mean()
    chi2_train = (z[ALA3.train_keys] ** 2).mean()
    chi2_test = (z[ALA3.test_keys] ** 2).mean()
    chi2[ff] = chi2_all, chi2_train, chi2_test
    data[ff] = predictions.mean()
    mcmc_filename = "mcmc_traces/mu_%s_%s_reg-%.1f-BB%d.h5"
    mu_mcmc = pd.concat([pd.HDFStore(mcmc_filename % (ff, prior, regularization_strength, bayesian_bootstrap_run))["data"] for bayesian_bootstrap_run in range(num_BB)])
    data["%s_%s" % (ff, prior)] = mu_mcmc.mean()
    z = (mu_mcmc - measurements) / uncertainties
    chi2_all = (z ** 2).mean().mean()
    chi2_train = (z[ALA3.train_keys] ** 2).mean().mean()
    chi2_test = (z[ALA3.test_keys] ** 2).mean().mean()
    chi2["%s_%s" % (ff, prior)] = chi2_all, chi2_train, chi2_test
    
data["Uncertainty"] = uncertainties
data["NMR"] = measurements
data = pd.DataFrame(data).T

index = ["all", "train", "test"]
chi2 = pd.DataFrame(chi2, index=index).T

data["all"] = chi2["all"]
data["train"] = chi2.train
data["test"] = chi2.test

print data.T.to_latex(float_format=(lambda x: "%.1f"%x))
