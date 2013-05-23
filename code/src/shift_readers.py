import pandas as pd
import string
import numpy as np

""" TO DO: implement shiftx2 parser."""

def read_sparta_tab(filename, skiprows):
    names = string.split("RESID RESNAME ATOMNAME SS_SHIFT SHIFT RC_SHIFT HM_SHIFT EF_SHIFT SIGMA")
    x = pd.io.parsers.read_table(filename, skiprows=skiprows, header=None, names=names, sep="\s*")
    return x

def format_sparta(dataframe):
    x = dataframe.rename(index=lambda i: "CS_%d_%s" % (dataframe.ix[i]["RESID"], dataframe.ix[i]["ATOMNAME"]))
    return x
    
def read_all_sparta(filenames, skiprows):
    num_frames = len(filenames)

    filename = filenames[0]
    frame = read_sparta_tab(filename, skiprows)
    x = format_sparta(frame)
    num_measurements = x.shape[0]
    
    d = pd.DataFrame(np.zeros((num_frames, num_measurements)), columns=x.index.values)
    for k, filename in enumerate(filenames):
        frame = read_sparta_tab(filename, skiprows)
        x = format_sparta(frame)
        d.iloc[k] = x["SHIFT"]

    return d
    
def read_ppm_data(filename):
    x = pd.io.parsers.read_table(filename, header=None, sep="\s*")
    res_id = x.iloc[:,0]
    res_name = x.iloc[:,1]
    atom_name = x.iloc[:,2]
    values = x.iloc[:,4:].values
    indices = ["CS_%d_%s" % (res_id[i], atom_name[i]) for i in range(len(res_id))]
    d = pd.DataFrame(values.T, columns=indices)
    return d

def read_shiftx2_intermediate(directory):
    atom_name = np.loadtxt(directory + "/shifts_atoms.txt", "str")
    res_id = np.loadtxt(directory + "/shifts_resid.dat", 'int')
    values = np.load(directory + "/shifts.npz")["arr_0"]
    indices = ["CS_%d_%s" % (res_id[i], atom_name[i]) for i in range(len(res_id))]
    d = pd.DataFrame(values, columns=indices)
    return d

def read_shiftx2(filename):
    dataframe = pd.io.parsers.read_csv(filename)
    x = dataframe.rename(index=lambda i: "CS_%d_%s" % (dataframe.ix[i]["NUM"], dataframe.ix[i]["ATOMNAME"]))
    return x

def read_all_shiftx2(filenames):
    num_frames = len(filenames)

    filename = filenames[0]
    x = read_shiftx2(filename)
    num_measurements = x.shape[0]
    
    d = pd.DataFrame(np.zeros((num_frames, num_measurements)), columns=x.index.values)
    for k, filename in enumerate(filenames):
        x = read_shiftx2(filename)
        d.iloc[k] = x["SHIFT"]

    return d
