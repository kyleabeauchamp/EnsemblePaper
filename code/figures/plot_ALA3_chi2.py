import experiment_loader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import ALA3

prior = "maxent"
n = len(ALA3.ff_list)

chi2 = np.zeros(n)
chi2_raw = np.zeros(n)
for k, ff in enumerate(ALA3.ff_list):
    model_directory = ALA3.data_dir + "/%s/models-%s/" % (ff, ALA3.model)
    chi2[k] = np.loadtxt(model_directory + "reg-%d-chi2_MCMC.dat" % ALA3.regularization_strength_dict[prior][ff])
    chi2_raw[k] = np.loadtxt(model_directory + "chi2_raw.dat")

x_local = np.arange(3)
x_global = 4*np.arange(n)
ylim_list = np.array([[0.1,1.0],[0.05,1.0],[0.002,0.4]])
ylim = np.array([0.05,20.0])

plt.bar(x_local[0] + x_global, chi2_raw, color='b', label="MD", log=True)
plt.bar(x_local[1] + x_global, chi2, color='g', label="LVBP", log=True)

plt.xticks(x_global + 1, ALA3.ff_list, rotation=60, fontsize=10)
plt.ylabel("Reduced $\chi^2$")

plt.legend(loc=0)
plt.title("Reduced $\chi^2$ by force field")
plt.ylim(ylim)
plt.xlim(-0.5,x_global.max() + x_local.max() + 1.5)
#plt.savefig(ALA3.outdir+"/ALA3_chi2.pdf",bbox_inches='tight')
