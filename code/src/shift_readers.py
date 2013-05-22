import pandas as pd
import string
import numpy as np

""" TO DO: implement shiftx2 parser."""

def read_sparta_tab(filename):
    names = string.split("RESID RESNAME ATOMNAME SS_SHIFT SHIFT RC_SHIFT HM_SHIFT EF_SHIFT SIGMA")
    x = pd.io.parsers.read_table(filename, skiprows=27, header=None, names=names, sep="\s*")
    return x

def format_sparta(dataframe):
    x = dataframe.rename(index=lambda i: "CS-%d-%s" % (dataframe.ix[i]["RESID"], dataframe.ix[i]["ATOMNAME"]))
    return x
    
def read_all_sparta(filenames):
    num_frames = len(filenames)

    filename = filenames[0]
    frame = read_sparta_tab(filename)
    x = format_sparta(frame)
    num_measurements = x.shape[0]
    
    d = pd.DataFrame(np.zeros((num_frames, num_measurements)), columns=x.index.values)
    for i in xrange(num_frames):
        frame = read_sparta_tab(filename)
        x = format_sparta(frame)
        d.iloc[i] = x["SHIFT"]

    return d
    
def read_ppm_data(filename):
    x = pd.io.parsers.read_table(filename, header=None, sep="\s*")
    res_id = x.iloc[:,0]
    res_name = x.iloc[:,1]
    atom_name = x.iloc[:,2]
    values = x.iloc[:,4:].values
    indices = ["CS-%d-%s" % (res_id[i], atom_name[i]) for i in range(len(res_id))]
    d = pd.DataFrame(values.T, columns=indices)
    return d

def read_shiftx2(directory):
    atom_name = np.loadtxt(directory + "/shifts_atoms.txt", "str")
    res_id = np.loadtxt(directory + "/shifts_resid.dat", 'int')
    values = np.load(directory + "/shifts.npz")["arr_0"]
    indices = ["CS-%d-%s" % (res_id[i], atom_name[i]) for i in range(len(res_id))]
    d = pd.DataFrame(values, columns=indices)
    return d
