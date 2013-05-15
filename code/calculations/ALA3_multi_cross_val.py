import experiment_loader
import ALA3
import numpy as np
from fit_ensemble import lvbp

ff = "amber96"

directory = "/home/kyleb/dat/lvbp/%s/" % ff
out_dir = directory + "/models-%s/" % ALA3.model

regularization_strength_list = [1.0, 3.0, 5.0, 7.0]
stride_list = [40, 20, 10, 5, 1]
for stride in stride_list:
    for regularization_strength in regularization_strength_list:
        print(stride, regularization_strength)
        keys, measurements, predictions, uncertainties, phi, psi = experiment_loader.load(directory, stride=stride)
        
        bootstrap_index_list = np.array_split(np.arange(len(predictions)), ALA3.kfold)
        train_chi, test_chi = lvbp.cross_validated_mcmc(predictions, measurements, uncertainties, regularization_strength, bootstrap_index_list, ALA3.num_samples, thin=ALA3.thin)
        
        test_chi = test_chi.mean()
        train_chi = train_chi.mean()
        
        np.savetxt(out_dir+"/reg-%d-stride-%d-cross_val_score.%dfold.dat" % (regularization_strength, stride, ALA3.kfold), [train_chi,test_chi])