import pandas as pd
import numpy as np
import experiment_loader
import ALA3
from fitensemble import belt
import itertools

num_BB = 2
grid = itertools.product(ALA3.ff_list, ALA3.prior_list)

data = {}
for k, (ff, prior) in enumerate(grid):
    print(ff, prior)
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]
    predictions, measurements, uncertainties = experiment_loader.load(ff, keys=None)
    z = (predictions.mean() - measurements) / uncertainties
    chi2_all = (z ** 2).mean()
    chi2_train = (z[ALA3.train_keys] ** 2).mean()
    chi2_test = (z[ALA3.test_keys] ** 2).mean()
    data[ff] = chi2_all, chi2_train, chi2_test
    mcmc_filename = "mcmc_traces/mu_%s_%s_reg-%.1f-BB%d.h5"
    mu_mcmc = pd.concat([pd.HDFStore(mcmc_filename % (ff, prior, regularization_strength, bayesian_bootstrap_run))["data"] for bayesian_bootstrap_run in range(num_BB)])
    z = (mu_mcmc - measurements) / uncertainties
    chi2_all = (z ** 2).mean().mean()
    chi2_train = (z[ALA3.train_keys] ** 2).mean().mean()
    chi2_test = (z[ALA3.test_keys] ** 2).mean().mean()
    data["%s_%s" % (ff, prior)] = chi2_all, chi2_train, chi2_test
    
columns = ["all", "train", "test"]
data = pd.DataFrame(data, index=columns).T
print("***********")
print data.to_latex(float_format=(lambda x: "%.2f"%x))
