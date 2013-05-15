import experiment_loader
import ALA3
import numpy as np
from fit_ensemble import lvbp
import sys

ff = sys.argv[1]
regularization_strength = float(sys.argv[3])

directory = "/home/kyleb/dat/lvbp/%s/" % ff
out_dir = directory + "/models-%s/" % ALA3.model

keys, measurements, predictions, uncertainties, phi, psi = experiment_loader.load(directory, stride=ALA3.stride)


bootstrap_index_list = np.array_split(np.arange(len(predictions)), ALA3.kfold)
train_chi, test_chi = lvbp.cross_validated_mcmc(predictions, measurements, uncertainties, regularization_strength, bootstrap_index_list, ALA3.num_samples)

test_chi = test_chi.mean()
train_chi = train_chi.mean()

np.savetxt(out_dir+"/reg-%d-stride-%d-cross_val_score.%dfold.dat" % (regularization_strength, ALA3.kfold), [train_chi,test_chi])