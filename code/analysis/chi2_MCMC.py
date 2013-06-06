import numpy as np
import experiment_loader
import ALA3
from fitensemble import lvbp
import sys

#ff = "charmm27"
#prior = "maxent"
ff = sys.argv[1]
prior = sys.argv[2]

regularization_strength = ALA3.regularization_strength_dict[prior][ff]

directory = "%s/%s" % (ALA3.data_dir , ff)
predictions, measurements, uncertainties = experiment_loader.load(directory)
model_directory = directory + "/models-maxent/"
lvbp_model = lvbp.LVBP.load(model_directory + "/reg-%d-BB0.h5" % regularization_strength)

reduced_chi2 = []
for pi in lvbp_model.iterate_populations():
    mu = predictions.T.dot(pi)
    reduced_chi2.append((((mu - measurements) / uncertainties)**2).mean(0))

mu0 = predictions.mean(0)
chi2_raw = (((mu0 - measurements) / uncertainties)**2).mean(0)

np.savetxt(model_directory + "reg-%d-chi2_MCMC_train.dat" % ALA3.regularization_strength_dict[prior][ff], reduced_chi2)


predictions, measurements, uncertainties = experiment_loader.load(directory, keys=ALA3.test_keys)
reduced_chi2 = []
for pi in lvbp_model.iterate_populations():
    mu = predictions.T.dot(pi)
    reduced_chi2.append((((mu - measurements) / uncertainties)**2).mean(0))

mu0 = predictions.mean(0)
chi2_raw = (((mu0 - measurements) / uncertainties)**2).mean(0)

np.savetxt(model_directory + "reg-%d-chi2_MCMC_test.dat" % ALA3.regularization_strength_dict[prior][ff], reduced_chi2)
