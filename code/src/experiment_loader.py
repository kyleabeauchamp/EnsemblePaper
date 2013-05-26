import numpy as np
import pandas as pd
import ALA3
from fitensemble.nmr_tools import scalar_couplings, chemical_shifts

def load_predictions(directory):
    shifts = pd.HDFStore(directory + "/observables/combined.h5")["data"]
    couplings = pd.HDFStore(directory + "/observables/scalar_couplings.h5")["data"]
    predictions = shifts.join(couplings)

    return predictions

def load_uncertainties(measurements):
    atoms = measurements.index.get_level_values("name")
    sig = chemical_shifts.atom_uncertainties.combine_first(scalar_couplings.uncertainties)  # Series containing BOTH JC and CS uncertainties, indexed by the "name" multiindex key
    uncertainties = pd.Series(sig[atoms].values, index=measurements.index)
    return uncertainties

def load_measurements():
    x = pd.io.parsers.read_csv(ALA3.experiment_filename)
    x = x.pivot_table(rows=["experiment","resid","name"])["value"]
    return x

def load(directory, stride=1, keys=ALA3.train_keys):
    predictions = load_predictions(directory)
    measurements = load_measurements()
    uncertainties = load_uncertainties(measurements)

    #Select only the keys that we requested
    measurements = measurements[keys].dropna()
    predictions = predictions[keys].dropna()
    uncertainties = uncertainties[keys].dropna()
    
    #Drop indices where our keys were invalid
    keys = predictions.columns.intersection(measurements.index).intersection(uncertainties.index)
    
    predictions = predictions[keys][::stride].copy()
    measurements = measurements[keys].copy()
    uncertainties = uncertainties[keys].copy()

    return predictions, measurements, uncertainties

def load_rama(directory, stride):
    phi, psi = np.load(directory + "/rama.npz")["arr_0"]
    ass_raw = schwalbe_couplings.assign(phi, psi)
    state_ind = np.array([ass_raw==i for i in xrange(4)])
    
    return phi[::stride], psi[::stride], ass_raw[::stride], state_ind[:, ::stride]

