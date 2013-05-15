import numpy as np 
from msmbuilder import Trajectory
from msmbuilder.geometry import dihedral

ff_list = ["amber96","amber99","amber99sbnmr-ildn","oplsaa","charmm27"]

for ff in ff_list:
    print(ff)
    directory = "/home/kyleb/dat/lvbp/%s/" % ff
    R = Trajectory.load_from_xtc([directory + "/production/trajout.xtc"],directory + "/final.pdb")    
    ind = dihedral.get_indices(R)
    di = dihedral.compute_dihedrals(R,ind).T    
    phi = di[0]
    psi = di[3]
    data = np.array([phi,psi])
    np.savez_compressed(directory + "/rama.npz", data)

"""
ff = "amber99"
R = Trajectory.load_from_xtc(["/home/kyleb/dat/lvbp/GA-%s/md/trajout.xtc"%ff],"/home/kyleb/dat/lvbp/GA-%s/equil/native.pdb"%ff)

ind = dihedral.get_indices(R)
di = dihedral.compute_dihedrals(R,ind)

io.saveh("/home/kyleb/dat/lvbp/GA-%s/rama.h5"%ff,di)
"""