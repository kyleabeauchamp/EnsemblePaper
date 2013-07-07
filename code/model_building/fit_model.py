import numpy as np
from fitensemble import belt, ensemble_fitter
import experiment_loader
import sys
import ALA3
belt.ne.set_num_threads(1)

def run(ff, prior, regularization_strength, bootstrap_index_list):
    pymc_filename = ALA3.data_directory + "/models/model_%s_%s_reg-%.1f-BB%d.h5" % (ff, prior, regularization_strength, bayesian_bootstrap_run)
    populations_filename = ALA3.data_directory + "/frame_populations/pops_%s_%s_reg-%.1f-BB%d.dat" % (ff, prior, regularization_strength, bayesian_bootstrap_run)

    predictions, measurements, uncertainties = experiment_loader.load(ff)

    num_frames, num_measurements = predictions.shape
    bootstrap_index_list = np.array_split(np.arange(num_frames), ALA3.num_blocks)

    if bayesian_bootstrap_run == 0:
        prior_pops = None
    else:
        prior_pops = ensemble_fitter.sample_prior_pops(num_frames, bootstrap_index_list)

    if prior == "maxent":
        model = belt.MaxEnt_BELT(predictions.values, measurements.values, uncertainties.values, regularization_strength, prior_pops=prior_pops)
    elif prior == "dirichlet":
        model = belt.Dirichlet_BELT(predictions.values, measurements.values, uncertainties.values, regularization_strength, prior_pops=prior_pops)
    elif prior == "MVN":
        model = belt.MVN_BELT(predictions.values, measurements.values, uncertainties.values, regularization_strength, prior_pops=prior_pops)

    model.sample(ALA3.num_samples, thin=ALA3.thin, burn=ALA3.burn, filename=pymc_filename)
    p = model.accumulate_populations()
    np.savetxt(populations_filename, p)

if __name__ == "__main__":
    ff = sys.argv[1]
    prior = sys.argv[2]
    regularization_strength = float(sys.argv[3])
    bayesian_bootstrap_run = int(sys.argv[4])

    run(ff, prior, regularization_strength, bayesian_bootstrap_run)
