import ALA3
import pandas as pd
import numpy as np
import shift_readers

columns = ['CS_2_C', 'CS_2_CA', 'CS_2_CB','CS_2_H', 'CS_2_HA', 'CS_2_HB', 'CS_2_HB2', 'CS_2_HB3', 'CS_2_N']

for ff in ALA3.ff_list:
    x_shiftx = pd.HDFStore(ALA3.data_dir + "/%s/observables/shiftx.h5" % ff, 'r')["data"]
    x_sparta = pd.HDFStore(ALA3.data_dir + "/%s/observables/sparta.h5" % ff, 'r')["data"]
    x_ppm = pd.HDFStore(ALA3.data_dir + "/%s/observables/ppm.h5" % ff, 'r')["data"]

    x_shiftx["expt"] = x_shiftx.index 
    x_shiftx["model"] = "shiftx"
    y_shiftx = x_shiftx.pivot_table(rows=["expt","model"])

    x_ppm["expt"] = x_ppm.index 
    x_ppm["model"] = "ppm"
    y_ppm = x_ppm.pivot_table(rows=["expt","model"])

    x_sparta["expt"] = x_sparta.index 
    x_sparta["model"] = "sparta"
    y_sparta = x_sparta.pivot_table(rows=["expt","model"])

    a = pd.concat((y_shiftx, y_ppm, y_sparta))
    ave = a.groupby(level=0).mean()[columns]
    ave.to_hdf(ALA3.data_dir + "/%s/observables/combined.h5" % ff, "data")
