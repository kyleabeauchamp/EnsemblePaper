import schwalbe_couplings
import numpy as np
import experiment_loader
import ALA3
from fitensemble import lvbp

ff = "charmm27"

directory = "/home/kyleb/dat/lvbp/%s/"%ff

keys, measurements, predictions, uncertainties, phi, psi = experiment_loader.load(directory)
ass_raw = schwalbe_couplings.assign(phi, psi)
state_ind = np.array([ass_raw==i for i in xrange(4)])

model_directory = "/home/kyleb/dat/lvbp/%s/models-%s/" % (ff, ALA3.model)

lvbp_model = lvbp.MaxEnt_LVBP.load(model_directory + "/reg-%d-BB0.h5" % ALA3.regularization_strength_dict[ff])
p = np.loadtxt(model_directory + "/reg-%d-frame-populations.dat" % ALA3.regularization_strength_dict[ff])

chi2 = []
for pi in lvbp_model.iterate_populations():
    mu = predictions.T.dot(pi)
    chi2.append((((mu - measurements) / uncertainties)**2).mean(0))

chi2_MCMC = np.mean(chi2)
mu = predictions.T.dot(p)
chi2_AVE = (((mu - measurements) / uncertainties)**2).mean(0)

mu0 = predictions.mean(0)
chi2_raw = (((mu0 - measurements) / uncertainties)**2).mean(0)

np.savetxt(model_directory + "reg-%d-chi2_MCMC.dat" % ALA3.regularization_strength_dict[ff], [chi2_MCMC])
np.savetxt(model_directory + "reg-%d-chi2_MAP.dat" % ALA3.regularization_strength_dict[ff], [chi2_AVE])
np.savetxt(model_directory + "chi2_raw.dat", [chi2_raw])