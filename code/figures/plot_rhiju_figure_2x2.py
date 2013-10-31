import ALA3
from fitensemble.nmr_tools import scalar_couplings
import matplotlib.pyplot as plt
import numpy as np
import experiment_loader
import matplotlib
from fitensemble import belt
matplotlib.rcParams.update({'font.size': 20})

num_bins = 100
num_bins_gauss = 200
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

ai = model.mcmc.trace("alpha")[:]
mu = model.trace_observable(predictions.values[:,0])

n = len(predictions)
prior_pops = np.ones(n) / float(n)

sigma = uncertainties.values[0]
val = measurements.values[0]


ymax_JC = 0.75
figure()

ax2 = subplot(221)

d1 = np.random.normal(-163.8, 4, size=n)
#d2 = np.random.normal(-76.0, 5, size=n)
#d3 = np.concatenate((np.random.normal(32, 9, size=n / 2), np.random.normal(-10, 9, size=n / 2)))
d3 = np.concatenate((np.random.normal(-76, 15, size=n * (6. / 10.)), np.random.normal(32, 15, size=n * (3. / 10.)), np.random.normal(0, 25, size=n * (1 / 10.))))
d4 = np.random.normal(88, 8, size=n)
d5 = np.random.normal(-76.0, 8, size=n)

hist(d5, bins=num_bins_gauss, color='purple', normed=True, histtype='step')
hist(d1, bins=num_bins_gauss, color='magenta', normed=True, histtype='step')
#hist(d2, bins=num_bins_gauss, color='purple', normed=True, histtype='step')
hist(d3, bins=num_bins_gauss, color='cyan', normed=True, histtype='step')
hist(d4, bins=num_bins_gauss, color='green', normed=True, histtype='step')


setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(-180, 180)
xticks([-180, -90, 0, 90, 180])
ylabel("Population")
title("b")




ax2 = subplot(223)

p0 = np.ones(n) / n
hist(phi1, bins=num_bins, color='r', normed=True, weights=prior_pops, histtype='step')
#yticks([0, 0.05, 0.1])
#setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(-180, 180)
ylabel("Population")


locations = range(1000, 4500, 400)
for k, location in enumerate(locations):
    pops = belt.get_populations_from_alpha(ai[location], predictions.values, prior_pops)
    hist(phi1, bins=num_bins, weights=pops, histtype='step', color='b', alpha=0.5, normed=True) 
    setp(ax2.get_yticklabels(), visible=False)
    xlim(-180, 180)

#xticks([-180, 0, 180])
xticks([-180, -90, 0, 90, 180])
ylabel("Population")
xlabel(r"$\phi[^{\circ}]$")
title("d")

ax2 = subplot(222)

J1 = scalar_couplings.J3_HN_HA(d1)
#J2 = scalar_couplings.J3_HN_HA(d2)
J3 = scalar_couplings.J3_HN_HA(d3)
J4 = scalar_couplings.J3_HN_HA(d4)
J5 = scalar_couplings.J3_HN_HA(d5)

plot([J5.mean()] * 2, [0, 1], '--', color='purple')
plot([J1.mean()] * 2, [0, 1], '--', color='magenta')
#plot([J2.mean()] * 2, [0, 1], '--', color='orange')
plot([J3.mean()] * 2, [0, 1], '--', color='cyan')
plot([J4.mean()] * 2, [0, 1], '--', color='green')

#plot([val - sigma, val + sigma], [0, 1], color='k')
fill_betweenx([0, 1], [val - sigma] * 2, [val + sigma] * 2, color='k', alpha=0.25)


hist(J5, bins=num_bins_gauss, color='purple', normed=True, histtype='step')
hist(J1, bins=num_bins_gauss, color='magenta', normed=True, histtype='step')
#hist(J2, bins=num_bins_gauss, color='orange', normed=True, histtype='step')
hist(J3, bins=num_bins_gauss, color='cyan', normed=True, histtype='step')
hist(J4, bins=num_bins_gauss, color='green', normed=True, histtype='step')


setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(0, 11)
xticks([0, 5, 10])
title("c")
ylim(0, ymax_JC)
#ylabel("Population")

#plot([], [], color='k', label="NMR")
#plot([], [], color='r', label="MD")
#plot([], [], color='b', label="BELT")
#plot([], [], color='purple', label="Normal")
#legend(loc=0, fontsize="small")



ax2 = subplot(224)


#plot([val - sigma, val + sigma], [0, 0.48], color='k')
fill_betweenx([0, .5], [val - sigma] * 2, [val + sigma] * 2, color='k', alpha=0.25)

hist(predictions.values[:,0], bins=num_bins, color='r', normed=True, histtype='step')
mu0 = predictions.mean(0).values[0]
plot([mu0] * 2, [0, 0.5], color='r')
#setp(ax2.get_xticklabels(), visible=False)
setp(ax2.get_yticklabels(), visible=False)
xlim(0, 11)
#ylabel("Population")


locations = range(1000, 4500, 800)
for k, location in enumerate(locations):
    pops = belt.get_populations_from_alpha(ai[location], predictions.values, prior_pops)
    hist(predictions.values[:,0], bins=num_bins, weights=pops, normed=True, histtype='step', color='b', alpha=0.5) 
    mu_i = predictions.T.dot(pops).values[0]
    plot([mu_i] * 2, [0, 0.5], '--', color='b')
    setp(ax2.get_yticklabels(), visible=False)
    xlim(0, 11)


#ylabel("Population")
xlabel(r"J [Hz]")
xticks([0, 5, 10])
#ylim(0, ymax_JC)

title("e")
plt.savefig(ALA3.outdir + "/karplus_2x2.pdf", bbox_inches='tight')




figure()

plt.plot(phi, J, 'k', linewidth=8)
lower = yi - oi
upper = yi + oi
plt.fill_between(phi, lower * O, upper * O, color='k', alpha=alpha)
#plt.legend(loc=0, numpoints=1, scatterpoints=1, fontsize=7, fontsize='small')
yticks([0, 5, 10])
xticks([-180, -90, 0, 90, 180])
plt.xlim(-180,180)
plt.ylim(-0.5,10.5)

plot([], [], color='k', label="NMR")
plot([], [], color='r', label="MD")
plot([], [], color='b', label="BELT")
plot([], [], color='purple', label="Normal")
ylabel(r"J [Hz]")
xlabel(r"$\phi[^{\circ}]$")
title("a")

plt.savefig(ALA3.outdir + "/karplus_top_panel_karplus.pdf", bbox_inches='tight')

