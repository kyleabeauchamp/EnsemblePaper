import pandas as pd
import experiment_loader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import ALA3
import pandas as pd

ff_list = ALA3.ff_list
n = len(ALA3.ff_list)
bayesian_bootstrap_run = 0
use_log = True

d = pd.read_csv(ALA3.chi2_filename)

x = d[(d["method"] == "raw")&(d["prior"] == "maxent")]
chi2_raw = x.pivot(index="subset", columns="ff", values="value")

x_all = d[(d["subset"] == "test")&(d["method"] == "belt")]
chi2_all = x_all.pivot(index="prior",columns="ff",values="value")

x_test = d[(d["subset"] == "test")&(d["method"] == "belt")]
chi2_test = x_test.pivot(index="prior",columns="ff",values="value")

x_train = d[(d["subset"] == "train")&(d["method"] == "belt")]
chi2_train= x_train.pivot(index="prior",columns="ff",values="value")


x_local = np.arange(4)
x_global = 5 * np.arange(n)
ylim_list = np.array([[0.1,1.0],[0.05,1.0],[0.002,0.4]])
ylim = np.array([0.05,20.0])

plt.bar(x_local[0] + x_global, chi2_raw[ff_list].ix["all"], color='r', label="MD",log=use_log)
plt.bar(x_local[1] + x_global, chi2_all[ff_list].ix["maxent"], color='b', label="maxent",log=use_log)
plt.bar(x_local[2] + x_global, chi2_all[ff_list].ix["MVN"], color='g', label="MVN",log=use_log)

plt.xticks(x_global + 1, ff_list, rotation=60, fontsize=10)
plt.ylabel("Reduced $\chi^2$")

plt.legend(loc=0)
plt.title("Reduced $\chi^2$ by force field")
plt.ylim(ylim)
plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
plt.savefig(ALA3.outdir+"/chi2_all_priors.pdf",bbox_inches='tight')


plt.figure()
plt.bar(x_local[0] + x_global, chi2_raw[ff_list].ix["train"], color='r', label="MD",log=use_log)
plt.bar(x_local[1] + x_global, chi2_train[ff_list].ix["maxent"], color='b', label="maxent",log=use_log)
plt.bar(x_local[2] + x_global, chi2_train[ff_list].ix["MVN"], color='g', label="MVN",log=use_log)

plt.xticks(x_global + 1, ALA3.ff_list, rotation=60, fontsize=10)
plt.ylabel("Reduced $\chi^2$")

plt.legend(loc=0)
plt.title("Reduced $\chi^2$ by force field (Training Set)")
plt.ylim(ylim)
plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
plt.savefig(ALA3.outdir+"/chi2_train_priors.pdf",bbox_inches='tight')


plt.figure()
plt.bar(x_local[0] + x_global, chi2_raw[ff_list].ix["test"], color='r', label="MD",log=use_log)
plt.bar(x_local[1] + x_global, chi2_test[ff_list].ix["maxent"], color='b', label="maxent",log=use_log)
plt.bar(x_local[2] + x_global, chi2_test[ff_list].ix["MVN"], color='g', label="MVN",log=use_log)

plt.xticks(x_global + 1, ALA3.ff_list, rotation=60, fontsize=10)
plt.ylabel("Reduced $\chi^2$")

plt.legend(loc=0)
plt.title("Reduced $\chi^2$ by force field (Test Set)")
plt.ylim(ylim)
plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
plt.savefig(ALA3.outdir+"/chi2_test_priors.pdf",bbox_inches='tight')
