import ALA3
from fitensemble.nmr_tools import scalar_couplings
import matplotlib.pyplot as plt
import numpy as np
import experiment_loader
import matplotlib
from fitensemble import belt
matplotlib.rcParams.update({'font.size': 20})

alpha = 0.2
num_grid = 500
phi = np.linspace(-180,180,num_grid)
O = np.ones(num_grid)
colors = ["b","g","r","y"]

simulation_data = {}

ff = "amber99"  # Load FF data for comparison, not used in actual figure
phi1, psi1, ass_raw, state_ind = experiment_loader.load_rama(ff, 1)

J = scalar_couplings.J3_HN_HA(phi)

predictions, measurements, uncertainties = experiment_loader.load(ff, keys=[("JC", 2, "J3_HN_HA")])
yi = measurements.iloc[0]
oi = uncertainties.iloc[0]

factor = 1.0
regularization_strength = 0.2
model = belt.MaxEntBELT(predictions.values, measurements.values, factor * uncertainties.values, regularization_strength)
model.sample(5000)

obs = belt.

ai = model.mcmc.trace("alpha")[:]
mu = model.trace_observable(predictions.values[:,0])

n = len(predictions)
prior_pops = np.ones(n) / float(n)

ax1 = plt.subplot(311)
plt.plot(phi, J, 'b', label="Karplus Curve")
#plt.plot(phi,O*yi,"k", label="NMR", lw=.2)
plt.plot([], [], "k", label="NMR", alpha=alpha, lw=5)

plt.plot(phi,O * predictions.values.mean(),"k", label="MD")
plt.plot(phi,O * mu.mean(),"k--", label="BELT")
setp(ax1.get_xticklabels(), visible=False)
lower = yi - oi
upper = yi + oi
plt.fill_between(phi,lower*O,upper*O,color='k',alpha=alpha, label="NMR")
plt.ylabel(r"$f_i(\phi) = $ J")
plt.legend(loc=4, numpoints=1, scatterpoints=1, fontsize=7)
yticks([0, 5, 10])
plt.xlim(-180,180)
plt.ylim(-0.5,10.5)


ax2 = subplot(312)
p0 = np.ones(n) / n
hist(phi1, bins=100, color='k', normed=True, weights=prior_pops)
#yticks([0, 0.05, 0.1])
setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(-180, 180)
ylabel("MD")


locations = range(1000, 4500, 400)
ax2 = subplot(313)
for k, location in enumerate(locations):
    pops = belt.get_populations_from_alpha(ai[location], predictions.values, prior_pops)
    hist(phi1, bins=100, weights=pops, histtype='step', color='k', alpha=0.5) 
    setp(ax2.get_yticklabels(), visible=False)
    xlim(-180, 180)

ylabel("BELT")
xlabel(r"$\phi$")
plt.savefig(ALA3.outdir + "/karplus_three_panels.pdf", bbox_inches='tight')
