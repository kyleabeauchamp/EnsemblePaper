from fitensemble.nmr_tools import scalar_couplings
import matplotlib.pyplot as plt
import numpy as np
import experiment_loader
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

outdir = "/home/kyleb/src/pymd/Papers/maxent/figures/"

alpha = 0.2
num_grid = 500
phi = np.linspace(-180,180,num_grid)
O = np.ones(num_grid)
colors = ["b","g","r","y"]

simulation_data = {}

ff = "amber99"  # Load FF data for comparison, not used in actual figure
phi1, psi1, ass_raw, state_ind = experiment_loader.load_rama(ff, 1)

phi = np.linspace(-180, 180, 360)
psi = np.linspace(-180, 180, 360)

J = scalar_couplings.J3_HN_HA(phi)

plt.plot(phi,simulation_data["J3_HN_HA_2"],'b',label="Karplus Curve")
yi = experiment_loader.experimental_data["J3_HN_HA_2"]
oi = experiment_loader.sigma_dict["J3_HN_HA"]
plt.plot(phi,O*yi,"k",label="Measured Average")
lower = yi - oi
upper = yi + oi
plt.fill_between(phi,lower*O,upper*O,color='k',alpha=alpha)
plt.xlabel(r"$\phi$ [$\circ$]")
plt.ylabel(r"$f_i(\phi) = $ J")
plt.title("Projecting $\phi$ onto J Couplings")
plt.legend(loc=0)
plt.xlim(-180,180)
plt.ylim(-0.5,10.5)
plt.savefig(outdir+"/single_karplus.pdf",bbox_inches='tight')