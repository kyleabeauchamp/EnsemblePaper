import glob
import ALA3
import pandas as pd
import numpy as np
from fitensemble.nmr_tools import chemical_shift_readers

for ff in ALA3.ff_list:
    print(ff)
    filename = "/home/kyleb/dat/lvbp-trajectory-data/%s_production/ppm/bb_details.dat" % ff
    x_ppm = chemical_shift_readers.read_ppm_data(filename)
    x_ppm.to_hdf(ALA3.data_dir + "/%s/observables/ppm.h5" % ff, "data", mode='w')

    num_conf = len(glob.glob("/home/kyleb/dat/lvbp-trajectory-data/%s_production/pdbs/frame*_pred.tab" % ff))
    filenames = ["/home/kyleb/dat/lvbp-trajectory-data/%s_production/pdbs/frame%d_pred.tab" % (ff, i) for i in range(num_conf)]
    x_sparta = chemical_shift_readers.read_all_sparta(filenames, skiprows=27)
    y = x_sparta.T.reset_index()
    y["resid"] -= 1
    x_sparta = y.pivot_table(rows=["experiment","resid","name"])
    x_sparta = x_sparta.T
    x_sparta.to_hdf(ALA3.data_dir + "/%s/observables/sparta.h5" % ff, "data", mode='w')

    x_shiftx = chemical_shift_readers.read_shiftx2_intermediate("/home/kyleb/dat/ala_lvbp/%s/shiftx2/" % ff)
    x_shiftx.to_hdf(ALA3.data_dir + "/%s/observables/shiftx.h5" % ff, "data", mode='w')
