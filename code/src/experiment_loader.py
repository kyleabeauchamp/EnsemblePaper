import numpy as np
import schwalbe_couplings
import pandas as pd
import ALA3
import ALA3_data

def load_predictions(directory):
    shifts = pd.HDFStore(directory + "/observables/combined.h5")["data"]
    couplings = pd.HDFStore(directory + "/observables/scalar_couplings.h5")["data"]
    predictions = pd.concat((shifts, couplings))    

def load(directory, stride=1, keys=None):
    predictions = load_predictions(directory)
    measurements = ALA3_data.measurements
    uncertainties = ALA3.uncertainties
    
    if keys == None:
        keys = ALA3.train_keys
    
    predictions = predictions[select_keys][::stride].copy()
    measurements = measurements[keys]
    uncertainties = uncertainties[keys]
    
    return predictions, measurements, uncertainties

def load_rama(directory, stride):
    phi, psi = np.load(directory + "/rama.npz")["arr_0"]
    ass_raw = schwalbe_couplings.assign(phi, psi)
    state_ind = np.array([ass_raw==i for i in xrange(4)])
    
    return phi[::stride], psi[::stride], ass_raw[::stride], state_ind[:, ::stride]

