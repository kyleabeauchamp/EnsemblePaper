import experiment_loader
import ALA3
import numpy as np
from fitensemble import belt
import sys

belt.ne.set_num_threads(1)

def run(ff, prior, regularization_strength):
    predictions, measurements, uncertainties = experiment_loader.load(ff, stride=ALA3.cross_val_stride)
    if prior == "maxent":
        model_factory = lambda predictions, measurements, uncertainties: belt.MaxEnt_BELT(predictions, measurements, uncertainties, regularization_strength)
    else:
        precision = np.cov(predictions.values.T)
        model_factory = lambda predictions, measurements, uncertainties: belt.MVN_BELT(predictions, measurements, uncertainties, regularization_strength, precision=precision)

    bootstrap_index_list = np.array_split(np.arange(len(predictions)), ALA3.kfold)
    train_chi, test_chi = belt.cross_validated_mcmc(predictions.values, measurements.values, uncertainties.values, model_factory, bootstrap_index_list, ALA3.num_samples, thin=ALA3.thin)

    print regularization_strength, train_chi.mean(), test_chi.mean()
    F = open(ALA3.cross_val_filename, 'a')
    F.write("%s,%s,%f,%d,%d,%f,%f \n"% (ff, prior, regularization_strength, ALA3.cross_val_stride, ALA3.num_samples, train_chi.mean(), test_chi.mean()))
    F.close()

if __name__ == "__main__":
    ff = sys.argv[1]
    prior = sys.argv[2]
    regularization_strength = float(sys.argv[3])

    run(ff, prior, regularization_strength)
