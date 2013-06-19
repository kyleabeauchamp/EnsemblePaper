import ALA3
import numpy as np
import matplotlib.pyplot as plt
from fitensemble import belt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

num_frames = 10000000
z = np.random.normal(size=(num_frames, 1))
rho = np.random.random_integers(0,1,size=(num_frames, 1)) * 2 - 1
x = (z + 2) * rho
prior_pops = np.ones(num_frames) / float(num_frames)

num_bins = 50
use_log = False
scale = 0.5
ymax = 0.12

alpha = np.array([0.0])
p = belt.get_populations_from_alpha(alpha, x, prior_pops)
plt.hist(x, weights=p, bins=num_bins,color="b", log=use_log)
plt.title(r"$\alpha = 0$")
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
plt.ylim([0, ymax])
plt.savefig(ALA3.outdir+"/model_hist0.pdf",bbox_inches='tight')

alpha = np.array([-scale])
p = belt.get_populations_from_alpha(alpha, x, prior_pops)
plt.figure()
plt.hist(x,weights=p,bins=num_bins,color="g", log=use_log)
plt.title(r"$\alpha = %.1f$" % -scale)
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
plt.ylim([0, ymax])
plt.savefig(ALA3.outdir+"/model_hist-2.pdf",bbox_inches='tight')

alpha = np.array([scale])
p = belt.get_populations_from_alpha(alpha, x, prior_pops)
plt.figure()
plt.hist(x,weights=p,bins=num_bins, color="r", log=use_log)
plt.title(r"$\alpha = %.1f$" % scale)
plt.xlabel("Observable: $f(x)$")
plt.ylabel("Counts")
plt.ylim([0, ymax])
plt.savefig(ALA3.outdir+"/model_hist2.pdf",bbox_inches='tight')
