import ALA3
import numpy as np
import matplotlib.pyplot as plt

ff = "oplsaa"

si = np.load("./%s_two_expt.npz" % ff)["arr_0"]

mu = si.mean(0)[0]
ymin = 1
ymax = 1E5

plt.plot([mu, mu], [ymin, ymax], 'k')
plt.hist(si[:,0], bins=100, log=True)
plt.title("PPII Population of Sampled Ensembles")
plt.xlabel("PPII Population")
plt.ylabel("Number of Sampled Models")
plt.savefig(ALA3.outdir + "/%s_PPII_histogram.pdf" % ff, bbox_inches='tight')
