import numpy as np
import matplotlib.pyplot as plt
from fitensemble import lvbp
import matplotlib
matplotlib.rcParams.update({'font.size': 18})


outdir = "/home/kyleb/src/pymd/Papers/maxent/figures/"

num_frames = 10000000
x = np.random.normal(size=(num_frames, 1))
prior_pops = np.ones(num_frames) / float(num_frames)

num_bins = 50
use_log = True

alpha = np.array([0.0])
p = lvbp.get_populations_from_alpha(alpha, x, prior_pops)
plt.hist(x, weights=p, bins=num_bins,color="b", log=use_log)
plt.title(r"$\alpha = 0$")
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
#plt.savefig(outdir+"/model_hist0.pdf",bbox_inches='tight')

alpha = np.array([-2.0])
p = lvbp.get_populations_from_alpha(alpha, x, prior_pops)
plt.figure()
plt.hist(x,weights=p,bins=num_bins,color="g", log=use_log)
plt.title(r"$\alpha = -2$")
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
#plt.savefig(outdir+"/model_hist-2.pdf",bbox_inches='tight')

alpha = np.array([2.0])
p = lvbp.get_populations_from_alpha(alpha, x, prior_pops)
plt.figure()
plt.hist(x,weights=p,bins=num_bins, color="r", log=use_log)
plt.title(r"$\alpha = 2$")
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
#plt.savefig(outdir+"/model_hist2.pdf",bbox_inches='tight')
