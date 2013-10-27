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

n = len(predictions)
prior_pops = np.ones(n) / float(n)

figure()
ax1 = plt.subplot(221)
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

#plot([], [], color='k', label="NMR")
#plot([], [], color='r', label="MD")
#plot([], [], color='b', label="BELT")
#plot([], [], color='purple', label="Normal")
#legend(loc=0, fontsize="small")


plt.savefig(ALA3.outdir + "/karplus_top_panel_karplus.pdf", bbox_inches='tight')
