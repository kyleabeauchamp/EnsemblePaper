import glob
import numpy as np
from msmbuilder import io

ff = "charmm27"
in_directory = "/home/kyleb/dat/lvbp/trajectory_data/%s_production/pdbs/" % ff
out_directory = "/home/kyleb/dat/lvbp/%s/sparta/" % ff


num_confs = len(glob.glob("/%s/frame*.pdb" % in_directory))

all_shifts = []
all_errs = []

for i in xrange(num_confs):
    print(i)
    x = np.loadtxt("%s/frame%d_pred.tab"% (in_directory ,i),'str',skiprows=27)
    shifts = x[:,4].astype('float')
    errs = x[:,-1].astype('float')
    res_id = x[:,0].astype('int')
    atom_name = x[:,2]
    all_shifts.append(shifts)
    all_errs.append(errs)
    
    
all_shifts = np.array(all_shifts)
all_errs = np.array(all_errs)

mean_errs = all_errs.mean(0)
atom_name[atom_name == "HN"] = "H"

res_id -= 1

io.saveh("%s/shifts.h5" % out_directory,all_shifts)
np.savetxt("%s/shifts_errs.dat" % out_directory,mean_errs)
np.savetxt("%s/shifts_atoms.txt" % out_directory,atom_name,"%s")
np.savetxt("%s/shifts_resid.dat" % out_directory,res_id,"%d")

