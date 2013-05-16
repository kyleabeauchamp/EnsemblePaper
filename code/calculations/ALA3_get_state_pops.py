import numpy as np
import experiment_loader
import ALA3
from fitensemble import lvbp

ff = "amber99"
directory = ALA3.data_dir + "/%s/" % ff

predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory)

model_directory = "%s/models-%s/" % (directory, ALA3.model)

lvbp_model = lvbp.LVBP.load(model_directory + "/reg-%d-BB0.h5" % ALA3.regularization_strength_dict[ff])
state_pops_trace = lvbp_model.trace_observable(state_ind.T)

state_pops = state_pops_trace.mean(0)
state_uncertainties = state_pops_trace.std(0)

np.savetxt(model_directory + "reg-%d-state_populations.dat" % ALA3.regularization_strength_dict[ff], state_pops)
np.savetxt(model_directory + "reg-%d-state_uncertainties.dat" % ALA3.regularization_strength_dict[ff], state_uncertainties)
np.savez_compressed?(model_directory + "reg-%d-state_pops_trace.npz" % ALA3.regularization_strength_dict[ff], state_pops_trace)
