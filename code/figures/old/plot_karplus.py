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

ff = "amber96"  # Load FF data for comparison, not used in actual figure
phi1, psi1, ass_raw, state_ind = experiment_loader.load_rama(ff, 1)

J = scalar_couplings.J3_HN_HA(phi)

predictions, measurements, uncertainties = experiment_loader.load(ff, keys=[("JC", 2, "J3_HN_HA")])
yi = measurements.iloc[8]
oi = uncertainties.iloc[8]

plt.plot(phi, J, 'b',label="Karplus Curve")
#yi = experiment_loader.experimental_data["J3_HN_HA_2"]
#oi = experiment_loader.sigma_dict["J3_HN_HA"]
plt.plot(phi,O*yi,"k",label="NMR Measurement")
lower = yi - oi
upper = yi + oi
plt.fill_between(phi,lower*O,upper*O,color='k',alpha=alpha)
plt.xlabel(r"$\phi$ [$\circ$]")
plt.ylabel(r"$f_i(\phi) = $ J")
plt.title("Projecting $\phi$ onto J Couplings")
plt.legend(loc=0)
plt.xlim(-180,180)
plt.ylim(-0.5,10.5)
plt.savefig(ALA3.outdir+"/single_karplus.pdf",bbox_inches='tight')

factor = 2.5
regularization_strength = 0.2
model = belt.MaxEntBELT(predictions.values, measurements.values, factor * uncertainties.values, regularization_strength)
model.sample(5000)

ai = model.mcmc.trace("alpha")[:]

n = len(predictions)
prior_pops = np.ones(n) / float(n)

locations = [1000, 2000, 3000, 4000]
ax1 = subplot(413)
for k, location in enumerate(locations):
    pops = belt.get_populations_from_alpha(ai[location], predictions.values, prior_pops)
    ax2 = plt.subplot(4, 1, k + 1, sharex=ax1)
    hist(phi1, bins=100, weights=pops) 
    if k != 3:
        setp(ax2.get_xticklabels(), visible=False)
    if k == 0:
        title("Population Density")
    yticks([0, 0.05, 0.1])
    xlim(-180, 180)
    ylim(0, 0.1)

xlabel(r"$\phi$")
plt.savefig(ALA3.outdir + "/four_phi_histograms.pdf", bbox_inches='tight')
