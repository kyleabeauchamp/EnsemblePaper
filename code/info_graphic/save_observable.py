import numpy as np
import scipy.stats  #Note no longe resi 0 but resi 1
import ALA3, experiment_loader
import mdtraj

ff = "amber96"
prior = "maxent"
bayesian_bootstrap_run_list = [0,1]

regularization_strength = ALA3.regularization_strength_dict[prior][ff]

predictions, measurements, uncertainties = experiment_loader.load(ff)
phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)

frame1 = plt.figure()
hist(predictions[("CS", 2, "H")], bins=100)
#ylabel("Counts")
xlabel("H Chemical Shift [Hz]")

frame1.axes[0].get_yaxis().set_visible(False)
xlim(5.0, 10.0)

plt.savefig(ALA3.outdir + "/info_graphic/observable.pdf", bbox_inches='tight')


