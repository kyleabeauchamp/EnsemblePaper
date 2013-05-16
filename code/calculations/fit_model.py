import numpy as np
from fitensemble import lvbp
import experiment_loader
import sys
import ALA3

def run(ff, prior, regularization_strength, bootstrap_index_list):
    directory = "%s/%s" % (ALA3.data_dir , ff)
    out_dir = directory + "/models-%s/" % prior
    pymc_filename = out_dir + "/reg-%d-BB%d.h5" % (regularization_strength, bayesian_bootstrap_run)

    predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory, stride=ALA3.cross_val_stride)

    num_frames, num_measurements = predictions.shape
    bootstrap_index_list = np.array_split(np.arange(num_frames), ALA3.kfold)

    if bayesian_bootstrap_run == 0:
        prior_pops = None
    else:
        prior_pops = ensemble_fitter.sample_prior_pops(num_frames, bootstrap_index_list)

    if prior == "maxent":
        model = lvbp.MaxEnt_LVBP(predictions.values, measurements.values, uncertainties.values, regularization_strength, prior_pops=prior_pops)
    else:
        model = lvbp.MVN_LVBP(predictions, measurements, uncertainties, regularization_strength, prior_pops=prior_pops)

    model.sample(ALA3.num_samples, thin=ALA3.thin, burn=ALA3.burn, filename=pymc_filename)
    p = model.accumulate_populations()

    np.savetxt(out_dir+"/reg-%d-frame-populations.dat" % (regularization_strength), p)

if __name__ == "__main__":
    ff = sys.argv[1]
    prior = sys.argv[2]
    regularization_strength = float(sys.argv[3])
    bayesian_bootstrap_run = int(sys.argv[4])

    run(ff, prior, regularization_strength, bayesian_bootstrap_run)
