import glob
import ALA3
import pandas as pd
import numpy as np
import shift_readers

for ff in ALA3.ff_list:
    print(ff)
    filename = "/home/kyleb/dat/lvbp-trajectory-data/%s_production/ppm/bb_details.dat" % ff
    x_ppm = shift_readers.read_ppm_data(filename)
    x_ppm.to_hdf(ALA3.data_dir + "/%s/observables/ppm.h5" % ff, "data")

    num_conf = len(glob.glob("/home/kyleb/dat/lvbp-trajectory-data/%s_production/pdbs/frame*_pred.tab" % ff))
    filenames = ["/home/kyleb/dat/lvbp-trajectory-data/%s_production/pdbs/frame%d_pred.tab" % (ff, i) for i in range(num_conf)]
    x_sparta = shift_readers.read_all_sparta(filenames, skiprows=27)
    x_sparta.rename(columns=lambda x: x.replace("_2_","_1_").replace("_3_","_2_").replace("_4_","_3_"), inplace=True)
    x_sparta.rename(columns=lambda x: x.replace("HN","H"), inplace=True)
    x_sparta.to_hdf(ALA3.data_dir + "/%s/observables/sparta.h5" % ff, "data")

    x_shiftx = shift_readers.read_shiftx2_intermediate("/home/kyleb/dat/ala_lvbp/%s/shiftx2/" % ff)
    x_shiftx.to_hdf(ALA3.data_dir + "/%s/observables/shiftx.h5" % ff, "data")
