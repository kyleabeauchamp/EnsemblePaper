import pymc 
import numpy as np
import experiment_loader
import ALA3
from fitensemble import lvbp, ensemble_fitter

ff = "amber99sbnmr-ildn"
prior = "maxent"
directory = ALA3.data_dir + "/%s/" % ff
regularization_strength = ALA3.regularization_strength_dict[prior][ff]

predictions, measurements, uncertainties = experiment_loader.load(directory)
phi, psi, ass_raw, state_ind = experiment_loader.load_rama(directory, ALA3.stride)

ind = np.where(ass_raw == 0)[0]
z = ass_raw

data = []
num_blocks_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,35,50]
for num_blocks in num_blocks_list:
    dir_indices = np.array_split(np.arange(len(z)), num_blocks)
    prior_dirichlet = pymc.Dirichlet("prior_dirichlet", np.ones(num_blocks))
    
    @pymc.dtrm
    def populations(prior_dirichlet=prior_dirichlet):
        block_pops = np.zeros(num_blocks)
        block_pops[:-1] = prior_dirichlet  # The pymc Dirichlet does not explicitly store the final component
        block_pops[-1] = 1.0 - block_pops.sum()  # Calculate the final component from normalization.
        prior_populations = np.ones_like(z).astype('float')
        for k, ind in enumerate(dir_indices):
            prior_populations[ind] = block_pops[k] / len(ind)        
        return prior_populations        
    
    @pymc.dtrm
    def state_pops(populations=populations):
        return populations[ind].sum()
    
    populations.keep_trace = False    
    variables = [prior_dirichlet, populations, state_pops]
    
    S = pymc.MCMC(variables)
    S.sample(10000)
    
    state_pops_list = S.trace("state_pops")[:]
    data.append(state_pops_list)
    state_pops_list.mean(),state_pops_list.std()


data = np.array(data)
sig = data.std(1)
plt.plot(num_blocks_list, sig, 'o')

#block = 10 for, tau = 35 for amber99
#block ~15 for amber96
