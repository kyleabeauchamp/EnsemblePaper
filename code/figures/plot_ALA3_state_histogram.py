import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import ALA3

use_log = False

ff = "amber96"

directory = "/home/kyleb/dat/lvbp/%s/models-%s/" % (ff, ALA3.model)
mcmc_pops = np.loadtxt(directory + "reg-%d-state_pops_trace.dat" % ALA3.regularization_strength_dict[ff])

plt.hist(mcmc_pops[:,0],bins=50,normed=True)
plt.title(r"Posterior Populations: %s" % ff)
plt.xlabel(r"$PP_{II}$ Population")
plt.ylabel("# of MCMC Samples (Normalized)")
plt.savefig(ALA3.outdir + "/ALA3_%s_PPII_MCMC.pdf" % ff, bbox_inches='tight')