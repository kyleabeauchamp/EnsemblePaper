import pandas as pd
import experiment_loader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import ALA3
import pandas as pd

prior = "maxent"
n = len(ALA3.ff_list)
bayesian_bootstrap_run = 0

chi2 = np.zeros(n)
chi2_train = np.zeros(n)
chi2_test = np.zeros(n)
chi2_raw_train = np.zeros(n)
chi2_raw_test = np.zeros(n)
chi2_raw = np.zeros(n)
for k, ff in enumerate(ALA3.ff_list):
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]
    predictions, measurements, uncertainties = experiment_loader.load(ff, keys=None)
    mcmc_filename = "mcmc_traces/mu_%s_%s_reg-%.1f-BB%d.h5" % (ff, prior, regularization_strength, bayesian_bootstrap_run)
    mu_mcmc = pd.HDFStore(mcmc_filename)["data"]
    z = (measurements - mu_mcmc) / uncertainties
    chi2_train[k] = (z[ALA3.train_keys] ** 2.0).mean(0).mean()
    chi2_test[k] = (z[ALA3.test_keys] ** 2.0).mean(0).mean()
    chi2[k] = (z ** 2.0).mean(0).mean()
    z = (measurements - predictions.mean()) / uncertainties
    chi2_raw_train[k] = (z[ALA3.train_keys] ** 2).mean()
    chi2_raw_test[k] = (z[ALA3.test_keys] ** 2).mean()
    chi2_raw[k] = (z[ALA3.test_keys] ** 2).mean()

x_local = np.arange(3)
x_global = 4 * np.arange(n)
ylim_list = np.array([[0.1,1.0],[0.05,1.0],[0.002,0.4]])
ylim = np.array([0.05,20.0])

plt.bar(x_local[0] + x_global, chi2_raw, color='b', label="MD",log=True)
plt.bar(x_local[1] + x_global, chi2, color='g', label="BELT",log=True)

plt.xticks(x_global + 1, ALA3.ff_list, rotation=60, fontsize=10)
plt.ylabel("Reduced $\chi^2$")

plt.legend(loc=0)
plt.title("Reduced $\chi^2$ by force field")
plt.ylim(ylim)
plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
plt.savefig(ALA3.outdir+"/chi2_all.pdf",bbox_inches='tight')


plt.figure()
plt.bar(x_local[0] + x_global, chi2_raw_train, color='b', label="MD",log=True)
plt.bar(x_local[1] + x_global, chi2_train, color='g', label="BELT",log=True)

plt.xticks(x_global + 1, ALA3.ff_list, rotation=60, fontsize=10)
plt.ylabel("Reduced $\chi^2$")

plt.legend(loc=0)
plt.title("Reduced $\chi^2$ by force field (Training Set)")
plt.ylim(ylim)
plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
plt.savefig(ALA3.outdir+"/chi2_train.pdf",bbox_inches='tight')


plt.figure()
plt.bar(x_local[0] + x_global, chi2_raw_test, color='b', label="MD",log=True)
plt.bar(x_local[1] + x_global, chi2_test, color='g', label="BELT",log=True)

plt.xticks(x_global + 1, ALA3.ff_list, rotation=60, fontsize=10)
plt.ylabel("Reduced $\chi^2$")

plt.legend(loc=0)
plt.title("Reduced $\chi^2$ by force field (Test Set)")
plt.ylim(ylim)
plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
plt.savefig(ALA3.outdir+"/chi2_test.pdf",bbox_inches='tight')
