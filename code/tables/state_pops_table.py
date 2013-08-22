import pandas as pd
import numpy as np
import experiment_loader
import ALA3
from fitensemble import belt

BB_dict = {"amber96":[0,1],"amber99":[0,1],"amber99sbnmr-ildn":[0, 1],"charmm27":[0,1],"oplsaa":[0,1]}

mu = np.zeros((len(ALA3.ff_list), len(ALA3.prior_list), 4))
sig = np.zeros((len(ALA3.ff_list), len(ALA3.prior_list), 4))

for i,ff in enumerate(ALA3.ff_list):
    for j, prior in enumerate(ALA3.prior_list):
        print(ff, prior)
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        predictions, measurements, uncertainties = experiment_loader.load(ff)
        mcmc_pops = np.concatenate([np.load(ALA3.data_directory + "/state_populations/state_populations_%s_%s_reg_%.1f_BB%d.npz" % (ff, prior, regularization_strength, bayesian_bootstrap_run))["arr_0"] for bayesian_bootstrap_run in BB_dict[ff]])
        mu[i,j] = mcmc_pops.mean(0)
        sig[i,j] = mcmc_pops.std(0)

pops = pd.DataFrame(mu[:,:,0], index=ALA3.ff_list, columns=ALA3.prior_list)
print pops.to_latex(float_format=(lambda x: "%.2f"%x))

pops = pd.DataFrame(mu[:,:,1], index=ALA3.ff_list, columns=ALA3.prior_list)
print pops.to_latex(float_format=(lambda x: "%.2f"%x))

pops = pd.DataFrame(mu[:,:,2], index=ALA3.ff_list, columns=ALA3.prior_list)
print pops.to_latex(float_format=(lambda x: "%.2f"%x))

pops = pd.DataFrame(mu[:,:,3], index=ALA3.ff_list, columns=ALA3.prior_list)
print pops.to_latex(float_format=(lambda x: "%.2f"%x))


pops = pd.DataFrame(sig[:,:,0], index=ALA3.ff_list, columns=ALA3.prior_list)
print pops.to_latex(float_format=(lambda x: "%.2f"%x))

pops = pd.DataFrame(sig[:,:,1], index=ALA3.ff_list, columns=ALA3.prior_list)
print pops.to_latex(float_format=(lambda x: "%.2f"%x))

pops = pd.DataFrame(sig[:,:,2], index=ALA3.ff_list, columns=ALA3.prior_list)
print pops.to_latex(float_format=(lambda x: "%.2f"%x))

pops = pd.DataFrame(sig[:,:,3], index=ALA3.ff_list, columns=ALA3.prior_list)
print pops.to_latex(float_format=(lambda x: "%.2f"%x))


mcmc_pops = np.concatenate([np.load(ALA3.data_directory + "/state_populations/state_populations_%s_%s_reg_%.1f_BB%d.npz" % (ff, prior, ALA3.regularization_strength_dict[prior][ff], bayesian_bootstrap_run))["arr_0"] for bayesian_bootstrap_run in BB_dict[ff] for ff in ALA3.ff_list for prior in ALA3.prior_list])
