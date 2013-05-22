import glob
import ALA3
import pandas as pd
import numpy as np
import shift_readers

#for ff in ALA3.ff_list:
for ff in ["amber99sbnmr-ildn"]:
    print(ff)
    filename = "/home/kyleb/dat/lvbp-trajectory-data/%s_production/ppm/bb_details.dat" % ff
    x_ppm = shift_readers.read_ppm_data(filename)
    x_ppm.to_hdf(ALA3.data_dir + "/%s/observables/ppm.h5" % ff, "data")

    num_conf = len(glob.glob("/home/kyleb/dat/lvbp-trajectory-data/%s_production/pdbs/frame*_pred.tab" % ff))
    filenames = ["/home/kyleb/dat/lvbp-trajectory-data/%s_production/pdbs/frame%d_pred.tab" % (ff, i) for i in range(num_conf)]
    x_sparta = shift_readers.read_all_sparta(filenames)
    x_sparta.rename(columns=lambda x: x.replace("-2-","-1-").replace("-3-","-2-").replace("-4-","-3-"), inplace=True)
    x_sparta.rename(columns=lambda x: x.replace("HN","H"), inplace=True)
    x_sparta.to_hdf(ALA3.data_dir + "/%s/observables/sparta.h5" % ff, "data")

    x_shiftx = shift_readers.read_shiftx2("/home/kyleb/dat/ala_lvbp/%s/shiftx2/" % ff)
    x_shiftx.to_hdf(ALA3.data_dir + "/%s/observables/shiftx.h5" % ff, "data")
