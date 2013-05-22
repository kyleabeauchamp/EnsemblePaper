import pandas as pd
import numpy as np
import shift_readers

ff = "amber99"

filename = "/home/kyleb/dat/lvbp-trajectory-data/%s_production/ppm/bb_details.dat" % ff
x_ppm = shift_readers.read_ppm_data(filename)
x_ppm.rename(columns=lambda x: x.replace("-2-", "-3-"), inplace=True)

filenames = ["/home/kyleb/dat/lvbp-trajectory-data/%s_production/pdbs/frame%d_pred.tab" % (ff, i) for i in range(225001)]
x_sparta = shift_readers.read_all_sparta(filenames)

x_shiftx = shift_readers.read_shiftx2("/home/kyleb/dat/ala_lvbp/%s/shiftx2/" % ff)

ind = pd.MultiIndex.from_tuples([(c, "shiftx") for c in x_shiftx.columns])



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
ave = a.groupby(level=0).mean()
