import scipy.stats
import ALA3
import numpy as np
import matplotlib.pyplot as plt
from fitensemble import belt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

num_frames = 10000000
z = np.random.normal(size=(num_frames, 1))
rho = np.random.random_integers(0, 1, size=(num_frames, 1)) * 2 - 1
x = 0.5 * (z + 2) * rho
prior_pops = np.ones(num_frames) / float(num_frames)

num_bins = 50
use_log = False
scale = 1.0
ymin = 1E-5
ymax = 0.12

num_grid = 100
grid = np.linspace(-7, 7, num_grid)

alpha = np.array([0.0])
p = belt.get_populations_from_alpha(alpha, x, prior_pops)
plt.hist(x, weights=p, bins=num_bins,color="b", log=use_log)
y = grid * alpha[0]
#plt.plot(grid, y, 'k')
plt.title(r"$\alpha_1 = 0$")
plt.xlabel("Observable: $f_1(x)$")
plt.ylabel("Population")
plt.ylim([ymin, ymax])
plt.savefig(ALA3.outdir+"/model_hist0.pdf",bbox_inches='tight')

alpha = np.array([-scale])
p = belt.get_populations_from_alpha(alpha, x, prior_pops)
plt.figure()
plt.hist(x,weights=p,bins=num_bins,color="g", log=use_log)
y = grid * alpha[0]
#plt.plot(grid, y, 'k')
plt.title(r"$\alpha_1 = %d$" % -scale)
plt.xlabel("Observable: $f_1(x)$")
plt.ylabel("Population")
plt.ylim([ymin, ymax])
plt.savefig(ALA3.outdir+"/model_hist%d.pdf" % alpha[0],bbox_inches='tight')

alpha = np.array([scale])
p = belt.get_populations_from_alpha(alpha, x, prior_pops)
plt.figure()
plt.hist(x,weights=p,bins=num_bins, color="r", log=use_log)
y = grid * alpha[0]
#plt.plot(grid, y, 'k')
plt.title(r"$\alpha_1 = %d$" % scale)
plt.xlabel("Observable: $f_1(x)$")
plt.ylabel("Population")
plt.ylim([ymin, ymax])
plt.savefig(ALA3.outdir+"/model_hist%d.pdf" % alpha[0],bbox_inches='tight')


FE_ymin = 2
FE_ymax = 6
num_grid = 300
kt = 0.593
kde = scipy.stats.kde.gaussian_kde(x.T)
grid = np.linspace(-4, 4, num_grid)
p0 = kde.evaluate(np.array([grid]))

alpha = np.array([0.0])
p = belt.get_populations_from_alpha(alpha, np.array([grid]).T, p0)
FE = -kt * np.log(p)

plt.figure()
plt.plot(grid, FE)
plt.title(r"$\alpha_1 = 0$")
plt.xlabel("Observable: $f_1(x)$")
plt.ylabel("Free Energy")
plt.ylim([FE_ymin, FE_ymax])
plt.savefig(ALA3.outdir + "/model_landscape%d.pdf" % alpha[0], bbox_inches='tight')

alpha = np.array([scale])
p = belt.get_populations_from_alpha(alpha, np.array([grid]).T, p0)
FE = -kt * np.log(p)

plt.figure()
plt.plot(grid, FE)
plt.title(r"$\alpha_1 = %d$" % scale)
plt.xlabel("Observable: $f_1(x)$")
plt.ylabel("Free Energy")
plt.ylim([FE_ymin, FE_ymax])
plt.savefig(ALA3.outdir + "/model_landscape%d.pdf" % alpha[0], bbox_inches='tight')


alpha = np.array([-scale])
p = belt.get_populations_from_alpha(alpha, np.array([grid]).T, p0)
FE = -kt * np.log(p)

plt.figure()
plt.plot(grid, FE)
plt.title(r"$\alpha_1 = %d$" % -scale)
plt.xlabel("Observable: $f_1(x)$")
plt.ylabel("Free Energy")
plt.ylim([FE_ymin, FE_ymax])
plt.savefig(ALA3.outdir + "/model_landscape%d.pdf" % alpha[0], bbox_inches='tight')
