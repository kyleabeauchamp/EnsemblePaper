import schwalbe_couplings
import numpy as np
import experiment_loader
import ALA3
from fitensemble import lvbp

ff = "amber96"

directory = "/home/kyleb/dat/lvbp/%s/"%ff

keys, measurements, predictions, uncertainties, phi, psi = experiment_loader.load(directory)
ass_raw = schwalbe_couplings.assign(phi, psi)
state_ind = np.array([ass_raw==i for i in xrange(4)])

directory = "/home/kyleb/dat/lvbp/%s/models-%s/" % (ff, ALA3.model)

lvbp_model = lvbp.MaxEnt_LVBP.load(directory + "/reg-%d-BB0.h5" % ALA3.regularization_strength_dict[ff])
state_pops_trace = lvbp_model.trace_observable(state_ind.T)

state_pops = state_pops_trace.mean(0)
state_uncertainties = state_pops_trace.std(0)

np.savetxt(directory + "reg-%d-state_populations.dat" % ALA3.regularization_strength_dict[ff], state_pops)
np.savetxt(directory + "reg-%d-state_uncertainties.dat" % ALA3.regularization_strength_dict[ff], state_uncertainties)
np.savetxt(directory + "reg-%d-state_pops_trace.dat" % ALA3.regularization_strength_dict[ff], state_pops_trace)