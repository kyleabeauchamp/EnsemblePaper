import pandas as pd
import experiment_loader
import numpy as np
import ALA3
import pandas as pd

F = open(ALA3.chi2_filename, 'a')

num_BB = 2

for ff in ALA3.ff_list:
    for prior in ALA3.prior_list:
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        predictions, measurements, uncertainties = experiment_loader.load(ff, keys=None)
        mcmc_filename = "mcmc_traces/mu_%s_%s_reg-%.1f-BB%d.h5"
        mu_mcmc = pd.concat([pd.HDFStore(mcmc_filename % (ff, prior, regularization_strength, bayesian_bootstrap_run))["data"] for bayesian_bootstrap_run in range(num_BB)])

        z = (measurements - mu_mcmc) / uncertainties
        chi2_train = (z[ALA3.train_keys] ** 2.0).mean(0).mean()
        chi2_test = (z[ALA3.test_keys] ** 2.0).mean(0).mean()
        chi2 = (z ** 2.0).mean(0).mean()

        z = (measurements - predictions.mean()) / uncertainties
        chi2_raw_train = (z[ALA3.train_keys] ** 2).mean()
        chi2_raw_test = (z[ALA3.test_keys] ** 2).mean()
        chi2_raw = (z ** 2).mean()

        F.write("all,belt,%s,%s,%f\n" % (ff, prior, chi2))
        F.write("train,belt,%s,%s,%f \n" % (ff, prior, chi2_train))
        F.write("test,belt,%s,%s,%f \n" % (ff, prior, chi2_test))
        F.write("all,raw,%s,%s,%f \n" % (ff, prior, chi2_raw))
        F.write("train,raw,%s,%s,%f \n" % (ff, prior, chi2_raw_train))
        F.write("test,raw,%s,%s,%f \n" % (ff, prior, chi2_raw_test))
        F.flush()

