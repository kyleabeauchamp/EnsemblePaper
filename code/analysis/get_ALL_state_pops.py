import numpy as np
import experiment_loader
import ALA3
from fitensemble import lvbp
import itertools
import sys

BB = 1
num_threads = 2
rank = int(sys.argv[1])
grid = itertools.product(["amber99sbnmr-ildn"],["maxent","MVN"])
#grid = itertools.product(ALA3.ff_list, ALA3.prior_list)

for k, (ff, prior) in enumerate(grid):
    if k % num_threads == rank:
        print(ff, prior)
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        directory = ALA3.data_dir + "/%s/" % ff
        predictions, measurements, uncertainties = experiment_loader.load(directory)
        phi, psi, ass_raw, state_ind = experiment_loader.load_rama(directory, ALA3.stride)
        model_directory = "%s/models-%s/" % (directory, prior)
        lvbp_model = lvbp.LVBP.load(model_directory + "/reg-%d-BB%d.h5" % (regularization_strength, BB))
        state_pops_trace = lvbp_model.trace_observable(state_ind.T)
        state_pops = state_pops_trace.mean(0)
        state_uncertainties = state_pops_trace.std(0)
        #np.savetxt(model_directory + "reg-%d-state_populations.dat" % ALA3.regularization_strength_dict[prior][ff], state_pops)
        #np.savetxt(model_directory + "reg-%d-state_uncertainties.dat" % ALA3.regularization_strength_dict[prior][ff], state_uncertainties)
        np.savez_compressed(model_directory + "reg-%d-state_pops_trace_BB%d.npz" % (ALA3.regularization_strength_dict[prior][ff], BB), state_pops_trace)
