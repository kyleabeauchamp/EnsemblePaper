import matplotlib.pyplot as plt
import numpy as np
import pymc
import fit_ensemble
import experiment_loader
import schwalbe_couplings

directory = "/home/kyleb/dat/lvbp/amber99/"
subsample = 1
skip_shifts = False
keys,f_exp,f_sim,sigma_vector,phi,psi = experiment_loader.load(directory)
ass_raw = schwalbe_couplings.assign(phi,psi)
state_ind = np.array([ass_raw==i for i in xrange(4)])
ind = np.where(ass_raw == 0)[0]
z = ass_raw

data = []
num_blocks_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,35,50]
for num_blocks in num_blocks_list:
    dir_indices = np.array_split(np.arange(len(z)),num_blocks)
    prior_dirichlet = pymc.Dirichlet("prior_dirichlet",np.ones(num_blocks))
    
    @pymc.dtrm
    def populations(prior_dirichlet=prior_dirichlet):
        return fit_ensemble.dirichlet_to_prior_pops(prior_dirichlet,dir_indices,len(z))
    
    @pymc.dtrm
    def state_pops(populations=populations):
        return populations[ind].sum()
    
    populations.keep_trace = False    
    variables = [prior_dirichlet,populations,state_pops]
    
    S = pymc.MCMC(variables)
    S.sample(10000)
    
    state_pops_list = S.trace("state_pops")[:]
    data.append(state_pops_list)
    state_pops_list.mean(),state_pops_list.std()
    
    
data = np.array(data)
sig = data.std(1)
plt.plot(num_blocks_list,sig,'o')

#block = 10 for, tau = 35 for amber99
#block ~15 for amber96