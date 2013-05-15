import schwalbe_couplings
import numpy as np
import experiment_loader
import sys
#import os

ff = sys.argv[1]

directory = "/home/kyleb/dat/lvbp/%s/"%ff
out_dir = directory + "/raw/"
#os.mkdir(out_dir)
print(out_dir)

keys,f_exp,f_sim,sigma_vector,phi,psi = experiment_loader.load(directory)

ass_raw = schwalbe_couplings.assign(phi,psi)
state_ind = np.array([ass_raw==i for i in xrange(4)])

bootstrap_index_list = np.array_split(np.arange(len(f_sim)),10)

correlation = 50.
n = 1.0 * len(ass_raw) / correlation
raw_pops = 1.0*np.bincount(ass_raw)
raw_pops /= raw_pops.sum()
raw_errs = np.sqrt(raw_pops*(1-raw_pops) / n)

D0 = (f_sim.mean(0) - f_exp) / sigma_vector
raw_rms = (D0**2).mean()**0.5

np.savetxt(out_dir+"resi_0_populations.raw.dat",raw_pops)
np.savetxt(out_dir+"resi_0_uncertainties.raw.dat",raw_errs)
np.savetxt(out_dir+"raw_rms.dat",[raw_rms])