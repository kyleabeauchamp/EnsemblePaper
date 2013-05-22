import experiment_loader
import ALA3
import pandas as pd
import numpy as np
import tables

"""Add the training data keys to the MCMC HDF5 files so they can be loaded by most recent version of FitEnsemble."""

cutoff = 1E-3
bayesian_bootstrap_run = 0

for ff in ALA3.ff_list:
    for prior in ALA3.prior_list:
        print(ff, prior)
        directory = "%s/%s" % (ALA3.data_dir , ff)
        out_dir = directory + "/models-%s/" % prior
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        pymc_filename = out_dir + "/reg-%d-BB%d.h5" % (regularization_strength, bayesian_bootstrap_run)
        predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, select_keys=ALA3.train_keys, stride=ALA3.stride)
        F = tables.File(pymc_filename,'a')
        p = F.root.predictions[:]
        measurement_indices = predictions.columns
        frame_indices = predictions.index
        if np.linalg.norm(predictions.values - p) < cutoff:
            print("GOOD")
            F.createCArray("/", "measurement_indices", tables.StringAtom(50), measurement_indices.shape)
            F.root.measurement_indices[:] = measurement_indices
            F.createCArray("/", "frame_indices", tables.Int64Atom(), frame_indices.shape)
            F.root.frame_indices[:] = frame_indices       
        else:
            print("error", ff, prior)
