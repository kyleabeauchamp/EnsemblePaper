import schwalbe_couplings
import pandas as pd
import numpy as np
from fitensemble import lvbp
import ALA3

for ff in ALA3.ff_list:
    directory = "%s/%s" % (ALA3.data_dir , ff)
    out_dir = directory + "/observables/"
    out_filename = out_dir + "/scalar_couplings.h5"

    phi, psi = np.load(directory + "/rama.npz")["arr_0"]

    simulation_data = {}
    simulation_data["J3_HN_HA_2"] = schwalbe_couplings.J3_HN_HA(phi)  
    simulation_data["J3_HN_Cprime_2"] = schwalbe_couplings.J3_HN_Cprime(phi)
    simulation_data["J3_HA_Cprime_2"] = schwalbe_couplings.J3_HA_Cprime(phi)
    simulation_data["J3_HN_CB_2"] = schwalbe_couplings.J3_HN_CB(phi)
    simulation_data["J1_N_CA_2"] = schwalbe_couplings.J1_N_CA(psi)
    simulation_data["J2_N_CA_3"] = schwalbe_couplings.J2_N_CA(psi)

    simulation_data = pd.DataFrame(simulation_data)

    simulation_data.to_hdf(out_filename, "data")
