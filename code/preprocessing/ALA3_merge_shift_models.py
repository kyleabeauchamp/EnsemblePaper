import numpy as np

shifts = np.load("./ppm/shifts.npz")["arr_0"]
atom_name = np.loadtxt("./ppm/shifts_atoms.txt",'str')
res_id = np.loadtxt("./ppm/shifts_resid.dat",'int')

ppm_data = {}
for k,(a,r) in enumerate(zip(atom_name,res_id)):
    print(k,a,r)
    key = (r,a)
    ppm_data[key] = shifts[:,k]


shifts = np.load("./sparta/shifts.npz")["arr_0"]
atom_name = np.loadtxt("./sparta/shifts_atoms.txt",'str')
res_id = np.loadtxt("./sparta/shifts_resid.dat",'int')

sparta_data = {}
for k,(a,r) in enumerate(zip(atom_name,res_id)):
    print(k,a,r)
    key = (r,a)
    sparta_data[key] = shifts[:,k]

shifts = np.load("./shiftx2/shifts.npz")["arr_0"]
atom_name = np.loadtxt("./shiftx2/shifts_atoms.txt",'str')
res_id = np.loadtxt("./shiftx2/shifts_resid.dat",'int')

shiftx2_data = {}
for k,(a,r) in enumerate(zip(atom_name,res_id)):
    print(k,a,r)
    key = (r,a)
    shiftx2_data[key] = shifts[:,k]

data = {}
for key,val in ppm_data.iteritems():
    if sparta_data.has_key(key):
        if shiftx2_data.has_key(key):
            v = (1.0 / 3.0) * (ppm_data[key] + sparta_data[key] + shiftx2_data[key])
            data[key] = v

            
keys = data.keys()
shifts = np.array([data[key] for key in keys]).T

res_id = np.array([key[0] for key in keys])
atom_name = np.array([key[1] for key in keys])
            
np.savez("./combined_shifts/shifts.npz",shifts)
np.savetxt("./combined_shifts/shifts_resid.dat",res_id,"%d")
np.savetxt("./combined_shifts/shifts_atoms.txt",atom_name,"%s")

