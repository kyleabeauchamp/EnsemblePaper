import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3

d = pd.io.parsers.read_table(ALA3.cross_val_filename, sep="\s*")

for ff, grp0 in d.groupby(["forcefield"]):
    plt.figure()
    for prior, grp1 in grp0.groupby(["prior"]):
        print(grp1.iloc[argmin(grp1["test_chi"])])
        plt.plot(grp1["regularization_strength"], grp1["test_chi"], 'o', label=prior)
    plt.title("Cross validation (%s)" % ff)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\chi^2$ (test)")
    plt.legend(loc='best')    
    plt.show()
