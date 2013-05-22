import ALA3
import pandas as pd
import numpy as np
import shift_readers

columns = ['CS-2-C', 'CS-2-CA', 'CS-2-CB','CS-2-H', 'CS-2-HA', 'CS-2-HB', 'CS-2-HB2', 'CS-2-HB3', 'CS-2-N']

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
