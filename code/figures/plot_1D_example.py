import numpy as np
import matplotlib.pyplot as plt
import fit_ensemble
import matplotlib
matplotlib.rcParams.update({'font.size': 18})


outdir = "/home/kyleb/src/pymd/Papers/maxent/figures/"

x = np.random.normal(size=(10000000,1))

num_bins = 50

alpha = np.array([0.0])
p = fit_ensemble.populations(alpha,x)
plt.hist(x,weights=p,bins=num_bins,color="b")
plt.title(r"$\alpha = 0$")
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
plt.savefig(outdir+"/model_hist0.pdf",bbox_inches='tight')

alpha = np.array([-2.0])
p = fit_ensemble.populations(alpha,x)
plt.figure()
plt.hist(x,weights=p,bins=num_bins,color="g")
plt.title(r"$\alpha = -2$")
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
plt.savefig(outdir+"/model_hist-2.pdf",bbox_inches='tight')

alpha = np.array([2.0])
p = fit_ensemble.populations(alpha,x)
plt.figure()
plt.hist(x,weights=p,bins=num_bins,color="r")
plt.title(r"$\alpha = 2$")
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
plt.savefig(outdir+"/model_hist2.pdf",bbox_inches='tight')