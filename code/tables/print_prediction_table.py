import pandas as pd
import numpy as np
import experiment_loader
import ALA3
from fitensemble import belt
import itertools

num_BB = 2
grid = itertools.product(ALA3.ff_list, ALA3.prior_list)

data = {}
sigma_data = {}
for k, (ff, prior) in enumerate(grid):
    print(ff, prior)
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]
    predictions, measurements, uncertainties = experiment_loader.load(ff, keys=None)
    data[ff] = predictions.mean()
    mcmc_filename = "mcmc_traces/mu_%s_%s_reg-%.1f-BB%d.h5"
    mu_mcmc = pd.concat([pd.HDFStore(mcmc_filename % (ff, prior, regularization_strength, bayesian_bootstrap_run))["data"] for bayesian_bootstrap_run in range(num_BB)])
    data["%s_%s" % (ff, prior)] = mu_mcmc.mean()
    sigma_data["%s_%s" % (ff, prior)] = mu_mcmc.std()

    
data["Uncertainty"] = uncertainties
data["NMR"] = measurements
data = pd.DataFrame(data).T
sigma_data = pd.DataFrame(sigma_data).T

print data.iloc[:, 0:4].to_latex(float_format=(lambda x: "%.2f"%x))
print data.iloc[:, 4:].to_latex(float_format=(lambda x: "%.2f"%x))

print sigma_data.iloc[:, 0:4].to_latex(float_format=(lambda x: "\pm %.2f"%x))
print sigma_data.iloc[:, 4:].to_latex(float_format=(lambda x: "\pm %.2f"%x))

bad_ind = []
for ff in ALA3.ff_list:
    for prior in ["MVN", "dirichlet"]:
        bad_ind.append("%s_%s" % (ff, prior))

data = data.ix[~np.in1d(data.index, bad_ind)]

print data.to_latex(float_format=(lambda x: "%.2f"%x))
