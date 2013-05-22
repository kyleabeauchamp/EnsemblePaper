import numpy as np
import schwalbe_couplings
import pandas as pd
import ALA3
import ALA3_data

def load_shift(atom_name,residue_id,all_atom_names,shift_data,residue_id_data):
    i = np.where((all_atom_names == atom_name) & (residue_id_data == residue_id))[0][0]    
    return shift_data[:,i]

def load_shifts(directory, simulation_data, model="combined_shifts"):
    d = np.load(directory + "/%s/shifts.npz" % model)["arr_0"]
    a = np.loadtxt(directory + "/%s/shifts_atoms.txt" % model,"str")
    rid = np.loadtxt(directory + "/%s/shifts_resid.dat" % model,'int')
    
    simulation_data["H_2"] = load_shift("H",2,a,d,rid)
    #simulation_data["HA_2"] = load_shift("HA",2,a,d,rid)  # Can't calculate HA in PPM, so skip
    #simulation_data["N_2"]  = load_shift("N",2,a,d,rid)  # N_2 predictions are all essentially the same.
    simulation_data["CA_2"] = load_shift("CA",2,a,d,rid)
    simulation_data["CB_2"] = load_shift("CB",2,a,d,rid)

def load_couplings(simulation_data, phi, psi):
    simulation_data["J3_HN_HA_2"] = schwalbe_couplings.J3_HN_HA(phi)  
    simulation_data["J3_HN_Cprime_2"] = schwalbe_couplings.J3_HN_Cprime(phi)
    simulation_data["J3_HA_Cprime_2"] = schwalbe_couplings.J3_HA_Cprime(phi)
    simulation_data["J3_HN_CB_2"] = schwalbe_couplings.J3_HN_CB(phi)
    simulation_data["J1_N_CA_2"] = schwalbe_couplings.J1_N_CA(psi)
    simulation_data["J2_N_CA_3"] = schwalbe_couplings.J2_N_CA(psi)
    
def load(directory, stride=1, select_keys=None):
    simulation_data = {}

    phi, psi = np.load(directory + "/rama.npz")["arr_0"]

    load_couplings(simulation_data, phi, psi)    
    load_shifts(directory,simulation_data)
        
    keys = experimental_data.keys()

    new_sigma_dict = dict([(key, sigma_dict[key[:-2]]) for key in keys])
    
    predictions = pd.DataFrame(simulation_data, columns=keys)[::stride].copy()
    measurements = pd.Series(experimental_data, index=keys)
    uncertainties = pd.Series(new_sigma_dict, index=keys)
    
    ass_raw = schwalbe_couplings.assign(phi, psi)
    state_ind = np.array([ass_raw==i for i in xrange(4)])
    
    if select_keys is None:
        select_keys = ALA3.train_keys
    
    predictions = predictions[select_keys].copy()
    measurements = measurements[select_keys].copy()
    uncertainties = uncertainties[select_keys].copy()
    
    return predictions, measurements, uncertainties, phi[::stride], psi[::stride], ass_raw[::stride], state_ind[:, ::stride]

