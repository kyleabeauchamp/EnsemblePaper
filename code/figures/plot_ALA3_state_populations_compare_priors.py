import numpy as np
import matplotlib.pyplot as plt
import scipy.stats  #Note no longe resi 0 but resi 1
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import ALA3

BB_dict = {"amber96":[0,1],"amber99":[0,1],"amber99sbnmr-ildn":[0],"charmm27":[0,1],"oplsaa":[0,1]}

use_log = False

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
mu_data_maxent = np.zeros((n,4))
err_data_maxent = np.zeros((n,4,2))
prior = "maxent"
for k,ff in enumerate(ALA3.ff_list):
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]
    mcmc_pops = np.concatenate([np.load(ALA3.data_directory + "/state_populations/state_populations_%s_%s_reg_%.1f_BB%d.npz" % (ff, prior, regularization_strength, bayesian_bootstrap_run))["arr_0"] for bayesian_bootstrap_run in BB_dict[ff]])
    for state, state_name in enumerate(state_name_list):
        y = mcmc_pops[:,state]
        mu_data_maxent[k,state] = y.mean()
        err_data_maxent[k,state,0] = y.mean() - scipy.stats.scoreatpercentile(y,16)
        err_data_maxent[k,state,1] = scipy.stats.scoreatpercentile(y,84) - y.mean()

mu_data_dirichlet = np.zeros((n,4))
err_data_dirichlet = np.zeros((n,4,2))
prior = "dirichlet"
for k,ff in enumerate(ALA3.ff_list):
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]
    mcmc_pops = np.concatenate([np.load(ALA3.data_directory + "/state_populations/state_populations_%s_%s_reg_%.1f_BB%d.npz" % (ff, prior, regularization_strength, bayesian_bootstrap_run))["arr_0"] for bayesian_bootstrap_run in BB_dict[ff]])
    for state, state_name in enumerate(state_name_list):
        y = mcmc_pops[:,state]
        mu_data_dirichlet[k,state] = y.mean()
        err_data_dirichlet[k,state,0] = y.mean() - scipy.stats.scoreatpercentile(y,16)
        err_data_dirichlet[k,state,1] = scipy.stats.scoreatpercentile(y,84) - y.mean()

mu_data_MVN = np.zeros((n,4))
err_data_MVN = np.zeros((n,4,2))
prior = "MVN"
for k,ff in enumerate(ALA3.ff_list):
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]
    mcmc_pops = np.concatenate([np.load(ALA3.data_directory + "/state_populations/state_populations_%s_%s_reg_%.1f_BB%d.npz" % (ff, prior, regularization_strength, bayesian_bootstrap_run))["arr_0"] for bayesian_bootstrap_run in BB_dict[ff]])
    for state, state_name in enumerate(state_name_list):
        y = mcmc_pops[:,state]
        mu_data_MVN[k,state] = y.mean()
        err_data_MVN[k,state,0] = y.mean() - scipy.stats.scoreatpercentile(y,16)
        err_data_MVN[k,state,1] = scipy.stats.scoreatpercentile(y,84) - y.mean()


x_local = np.arange(5)
x_global = 6 * np.arange(n)

ylim_list = np.array([[0.1,1.0],[0.05,1.0],[0.002,0.4]])
ylim = np.array([0.0,1.0])

for state, state_name in enumerate(state_name_list):
    plt.figure()
    raw_pops = all_raw_pops[:,state]
    maxent_pops = mu_data_maxent[:,state]
    dirichlet_pops = mu_data_dirichlet[:,state]
    mvn_pops = mu_data_MVN[:,state]
    errs_maxent = err_data_maxent[:,state].T
    errs_MVN = err_data_MVN[:,state].T
    plt.bar(x_local[0] + x_global,raw_pops,color='r',label="MD", log=use_log)
    plt.bar(x_local[1] + x_global,maxent_pops,color='b',label="maxent",yerr=errs_maxent,log=use_log,ecolor='k')
    plt.bar(x_local[2] + x_global,dirichlet_pops,color='c',label="dirichlet",yerr=errs_maxent,log=use_log,ecolor='k')    
    plt.bar(x_local[3] + x_global,mvn_pops,color='g',label="MVN",log=use_log,yerr=errs_MVN,ecolor="k")
    plt.xticks(x_global + 1, ALA3.ff_list,rotation=60,fontsize=10)
    plt.ylabel("%s Population"%state_name)
    plt.legend(loc=0,numpoints=1,scatterpoints=1)
    plt.title("%s Populations by Forcefield"%(state_name))
    plt.ylim(ylim)
    plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
    plt.savefig(ALA3.outdir+"/state_%d_by_forcefield_priors.pdf"%(state), bbox_inches='tight')
