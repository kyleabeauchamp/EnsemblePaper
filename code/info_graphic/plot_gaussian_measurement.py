import numpy as np
import scipy.stats  #Note no longe resi 0 but resi 1
import ALA3, experiment_loader
import mdtraj

ff = "amber96"
prior = "maxent"
bayesian_bootstrap_run_list = [0,1]

regularization_strength = ALA3.regularization_strength_dict[prior][ff]

predictions, measurements, uncertainties = experiment_loader.load(ff)

x0 = measurements[("CS", 2, "H")]
sigma = uncertainties[("CS", 2, "H")]

num_grid = 1000
grid = np.linspace(-8, 8, num_grid) * sigma + x0
y = np.exp(-0.5 * ((grid - x0) / sigma) ** 2.)

frame1 = plt.figure()
plt.plot(grid, y, 'k')
plt.plot([x0] * 2, [0, 1], 'k')
xlabel("H Chemical Shift [Hz]")
xlim(5.0, 10.0)
ylim(0, 1.1)

frame1.axes[0].get_yaxis().set_visible(False)

plt.savefig(ALA3.outdir + "/info_graphic/gaussian_measurement.pdf", bbox_inches='tight')


