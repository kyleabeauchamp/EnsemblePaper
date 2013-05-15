import string
import os
import glob
import numpy as np

num_conf= len(glob.glob("./frame*.pdb"))
flist = np.array(["./frame%d.pdb" % i for i in xrange(num_conf)])

ilist = np.arange(num_conf)

num_split = 500
split_flist = np.array_split(flist,num_split)
split_ilist = np.array_split(ilist,num_split)

for i,fi in enumerate(split_flist):
    flist_string = string.join(fi)
    cmd = "/home/kyleb/opt/Software/SPARTA+/bin/SPARTA+.linux -in %s -spartaDir ~/opt/Software/SPARTA+/" % (flist_string)
    os.system(cmd)
    for j in split_ilist[i]:
        os.remove("./frame%d_struct.tab" % j)