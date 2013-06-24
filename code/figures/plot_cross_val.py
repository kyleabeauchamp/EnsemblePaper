import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3

num_expt = 6.0

d = pd.io.parsers.read_csv(ALA3.cross_val_filename)

for ff, grp0 in d.groupby(["forcefield"]):
    plt.figure()
    for prior, grp1 in grp0.groupby(["prior"]):
        print(grp1.iloc[argmin(grp1["test_chi"])])
        plt.plot(grp1["regularization_strength"], grp1["test_chi"] / num_expt, 'o', label=prior)
    plt.title("Cross validation (%s)" % ff)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$n^{-1}\chi^2$")
    plt.legend(loc='best')
    plt.xlim(0.0, 15.0)
    plt.show()
    plt.savefig(ALA3.outdir + "/cross_val_%s.pdf" % ff, bbox_inches='tight')
