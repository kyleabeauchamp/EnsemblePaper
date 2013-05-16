import experiment_loader
import ALA3
import numpy as np
from fitensemble import lvbp
import sys

def run(ff, prior, regularization_strength):

    directory = "%s/%s" % (ALA3.data_dir , ff)
    out_dir = directory + "/cross_val/"

    predictions, measurements, uncertainties, phi, psi = experiment_loader.load(directory, stride=ALA3.cross_val_stride)

    if prior == "maxent":
        model_factory = lambda predictions, measurements, uncertainties: lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, regularization_strength)
    else:
        precision = np.cov(predictions.values.T)
        model_factory = lambda predictions, measurements, uncertainties: lvbp.MVN_LVBP(predictions, measurements, uncertainties, regularization_strength, precision=precision)

    bootstrap_index_list = np.array_split(np.arange(len(predictions)), ALA3.kfold)
    train_chi, test_chi = lvbp.cross_validated_mcmc(predictions.values, measurements.values, uncertainties.values, model_factory, bootstrap_index_list, ALA3.num_samples)

    test_chi = test_chi.mean()
    train_chi = train_chi.mean()
    print regularization_strength, train_chi.mean(), test_chi.mean()

    np.savetxt(out_dir+"/%s-reg-%d-stride%d-score.dat" % (prior, regularization_strength, ALA3.cross_val_stride), [train_chi, test_chi])

if __name__ == "__main__":
    ff = sys.argv[1]
    prior = sys.argv[2]
    regularization_strength = float(sys.argv[3])

    run(ff, prior, regularization_strength)
