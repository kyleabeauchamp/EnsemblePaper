from msmbuilder import Trajectory
import os
import numpy as np


r = Trajectory.load_from_xtc("../trajout.xtc","../pdbs/frame0.pdb")

a = r["AtomNames"]
a[a=="OT1"] = "O"
a[a=="OT2"] = "OXT"

r.save_to_pdb("traj.pdb")

cmd = """~/src/Software/ppm/ppm_linux_64.exe -pdb ./traj.pdb -mode detail"""
os.system(cmd)


x = np.loadtxt("./bb_details.dat",'str')
res_id = x[:,0].astype('int')
atom_name = x[:,2]
shifts = x[:,4:].astype('float').T

#os.mkdir("./ppm")
#np.savez_compressed("ppm/shifts.npz",  shifts)
#np.savetxt("ppm/shifts_atoms.txt", atom_name,"%s")
#np.savetxt("ppm/shifts_resid.dat", res_id,"%d")
