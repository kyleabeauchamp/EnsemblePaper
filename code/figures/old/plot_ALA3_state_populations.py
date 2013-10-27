import numpy as np
import matplotlib.pyplot as plt
import scipy.stats  #Note no longe resi 0 but resi 1
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import ALA3

BB_dict = {"amber96":[0],"amber99":[0],"amber99sbnmr-ildn":[0],"charmm27":[0],"oplsaa":[0]}

use_log = False
prior = "maxent"

n = len(ALA3.ff_list)

all_raw_pops = np.zeros((n, 4))
all_raw_errs = np.zeros((n, 4))
for k, ff in enumerate(ALA3.ff_list):
    directory = ALA3.data_directory + "/raw/"
    p0 = np.loadtxt(directory + "raw_populations_%s.dat" % ff)
    e0 = np.loadtxt(directory + "raw_uncertainties_%s.dat" % ff)  
    all_raw_pops[k] = p0
    all_raw_errs[k] = e0
        
state_name_list = ["PPII",r"$\beta$",r"$\alpha$"]
mu_data = np.zeros((n,4))
err_data = np.zeros((n,4,2))
for k,ff in enumerate(ALA3.ff_list):
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]
    mcmc_pops = np.concatenate([np.load(ALA3.data_directory + "/state_populations/state_populations_%s_%s_reg_%.1f_BB%d.npz" % (ff, prior, regularization_strength, bayesian_bootstrap_run))["arr_0"] for bayesian_bootstrap_run in BB_dict[ff]])
    for state, state_name in enumerate(state_name_list):
        y = mcmc_pops[:,state]
        mu_data[k,state] = y.mean()
        err_data[k,state,0] = y.mean() - scipy.stats.scoreatpercentile(y,16)
        err_data[k,state,1] = scipy.stats.scoreatpercentile(y,84) - y.mean()

x_local = np.arange(3)
x_global = 4*np.arange(n)
ylim_list = np.array([[0.1,1.0],[0.05,1.0],[0.002,0.4]])
ylim = np.array([0.0,1.0])
for state,state_name in enumerate(state_name_list):
    plt.figure()
    raw_pops = all_raw_pops[:,state]
    pops = mu_data[:,state]
    errs = err_data[:,state].T
    plt.bar(x_local[0] + x_global,raw_pops,color='b',label="MD",log=use_log)
    plt.bar(x_local[1] + x_global,pops,color='g',label="BELT",log=use_log,yerr=errs,ecolor="k")
    plt.xticks(x_global + 1, ALA3.ff_list,rotation=60,fontsize=10)
    plt.ylabel("%s Population"%state_name)
    plt.legend(loc=0)
    plt.title("%s Populations by Forcefield"%(state_name))
    plt.ylim(ylim)
    plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
    plt.savefig(ALA3.outdir+"/%s-state_%d_by_forcefield.pdf"%(prior, state), bbox_inches='tight')
