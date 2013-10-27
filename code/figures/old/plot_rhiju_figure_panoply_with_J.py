import ALA3
from fitensemble.nmr_tools import scalar_couplings
import matplotlib.pyplot as plt
import numpy as np
import experiment_loader
import matplotlib
from fitensemble import belt
matplotlib.rcParams.update({'font.size': 20})

alpha = 0.2
num_grid = 2000
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

#obs = belt.

ai = model.mcmc.trace("alpha")[:]
mu = model.trace_observable(predictions.values[:,0])

n = len(predictions)
prior_pops = np.ones(n) / float(n)



figure(figsize=(8,11))
ax1 = plt.subplot(421)
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


ax2 = subplot(423)

d1 = np.random.normal(-163.8, 5, size=n)
d2 = np.random.normal(-76.0, 5, size=n)
d3 = np.random.normal(32, 5, size=n)
d4 = np.random.normal(88, 5, size=n)

hist(d1, bins=100, color='k', normed=True, alpha=0.5)
hist(d2, bins=100, color='k', normed=True, alpha=0.5)
hist(d3, bins=100, color='k', normed=True, alpha=0.5)
hist(d4, bins=100, color='k', normed=True, alpha=0.5)
setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(-180, 180)
ylabel("Deg.")



ax2 = subplot(425)
p0 = np.ones(n) / n
hist(phi1, bins=100, color='k', normed=True, weights=prior_pops)
#yticks([0, 0.05, 0.1])
setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(-180, 180)
ylabel("MD")


locations = range(1000, 4500, 400)
ax2 = subplot(427)
for k, location in enumerate(locations):
    pops = belt.get_populations_from_alpha(ai[location], predictions.values, prior_pops)
    hist(phi1, bins=100, weights=pops, histtype='step', color='k', alpha=0.5) 
    setp(ax2.get_yticklabels(), visible=False)
    xlim(-180, 180)

xticks([-180, 0, 180])
ylabel("BELT")
xlabel(r"$\phi$")

ax2 = subplot(424)

J1 = scalar_couplings.J3_HN_HA(d1)
J2 = scalar_couplings.J3_HN_HA(d2)
J3 = scalar_couplings.J3_HN_HA(d3)
J4 = scalar_couplings.J3_HN_HA(d4)

hist(J1, bins=100, color='k', normed=True, alpha=0.5)
hist(J2, bins=100, color='k', normed=True, alpha=0.5)
hist(J3, bins=100, color='k', normed=True, alpha=0.5)
hist(J4, bins=100, color='k', normed=True, alpha=0.5)
setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(0, 11)
ylabel("Deg.")


ax2 = subplot(426)
p0 = np.ones(n) / n
hist(predictions.values[:,0], bins=100, color='k')
setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(0, 11)
ylabel("MD")


locations = range(1000, 4500, 400)
ax2 = subplot(428)
for k, location in enumerate(locations):
    pops = belt.get_populations_from_alpha(ai[location], predictions.values, prior_pops)
    hist(predictions.values[:,0], bins=100, weights=pops, histtype='step', color='k', alpha=0.5) 
    setp(ax2.get_yticklabels(), visible=False)
    xlim(0, 11)

ylabel("BELT")
xlabel(r"J [Hz]")
xticks([0, 5, 11])


plt.savefig(ALA3.outdir + "/karplus_eight_panels.pdf", bbox_inches='tight')
