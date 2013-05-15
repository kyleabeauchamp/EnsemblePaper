import numpy as np
from matplotlib import mlab
from msmbuilder import io

ff = "amber96"
#num_frames = 11250
num_frames = 225001
num_frames = 291622
num_frames = 295189
num_frames = 41250
data = []
for i in xrange(num_frames):
    print(i)
    d = mlab.csv2rec("%s/production/pdbs/frame%d.pdb.cs"%(ff,i))
    data.append(d["shift"])
    
data = np.array(data)

io.saveh("%s/shifts.h5"%ff,data)
np.savetxt("./%s/shifts_atoms.txt"%ff,d["atomname"],"%s")