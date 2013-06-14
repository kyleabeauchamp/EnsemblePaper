import ALA3
import numpy as np
import matplotlib.pyplot as plt

si = np.load("./amber99_two_expt.npz")["arr_0"]

hist(si[:,0], bins=100)
plt.title("PPII Population of Sampled Ensembles")
plt.xlabel("PPII Population")
plt.ylabel("Number of Sampled Models")
plt.savefig(ALA3.outdir + "/amber99_PPII_histogram.pdf", bbox_inches='tight')
