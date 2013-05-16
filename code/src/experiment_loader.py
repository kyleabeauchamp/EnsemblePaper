import numpy as np
import schwalbe_couplings
import pandas as pd

sigma_dict = {
"J3_HN_HA":0.36,"J3_HN_Cprime":0.30,"J3_HA_Cprime":0.24,"J3_Cprime_Cprime":0.13,"J3_HN_CB":0.22,"J1_N_CA":0.52659745254609414,"J2_N_CA":0.4776,
#"N":2.0862,"CA":0.7743,"CB":0.8583,"C":0.8699,"H":0.3783,"HA":0.1967         # ShiftX+ V1.07
"N":2.4625,"CA":0.7781,"CB":1.1760,"C":1.1309,"H":0.4685,"HA":0.2743,
}

experimental_data = {  # Numbering from Table S3 in Schwalbe
#"J3_HN_HA_2"        :5.68,  # Deleted because highly correlated to other measurements, r ranges from 0.71 to 0.997
"J3_HN_Cprime_2"    :1.13,
#"J3_HA_Cprime_2"    :1.84,  # Deleted because highly correlated to other measurements.  r > 0.7
"J3_HN_CB_2"        :2.39,
#"J1_N_CA_2"         :11.34,  # Deleted because highly correlated to other measurements.  r > 0.7
"J2_N_CA_3"         :8.45,

"H_2":8.571,
#"HA_2":4.355,
"CA_2":52.38,"CB_2":19.21
}

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
    
def load(directory, stride=1):
    simulation_data = {}

    phi, psi = np.load(directory + "/rama.npz")["arr_0"]

    load_couplings(simulation_data, phi, psi)    
    load_shifts(directory,simulation_data)
        
    keys = experimental_data.keys()

    #sigma_vector = np.array([sigma_dict[key[:-2]] for key in keys])
    new_sigma_dict = dict([(key, sigma_dict[key[:-2]]) for key in keys])
    #f_exp = np.array([experimental_data[key] for key in keys])
    #f_sim = np.array([simulation_data[key] for key in keys]).T
    
    predictions = pd.DataFrame(simulation_data, columns=keys)[::stride].copy()
    measurements = pd.Series(experimental_data, index=keys)
    uncertainties = pd.Series(new_sigma_dict, index=keys)
    
    ass_raw = schwalbe_couplings.assign(phi, psi)
    state_ind = np.array([ass_raw==i for i in xrange(4)])
    
    return predictions, measurements, uncertainties, phi[::stride], psi[::stride], ass_raw[::stride], state_ind[::stride]

    #return keys,f_exp,f_sim[::stride].copy(),sigma_vector, phi[::stride],psi[::stride]
