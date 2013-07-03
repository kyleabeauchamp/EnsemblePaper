import glob
import matplotlib.pyplot as plt
import numpy as np
import experiment_loader
import ALA3

predictions, measurements, uncertainties = experiment_loader.load("amber99", keys=None)
all_keys = measurements.index

for ff in ALA3.ff_list[::-1]:
    for i, key0 in enumerate(all_keys):
        for j, key1 in enumerate(all_keys):
            if i > j:
                print(ff, i, j, key0, key1)
                keys = [key0, key1]
                predictions, measurements, uncertainties = experiment_loader.load(ff, keys=keys)
                m0 = str(measurements.index[0])
                m1 = str(measurements.index[1])
                filename = "./pair_experiments/%s-%s-%s" % (ff, m0, m1)
                si = np.load(filename)["arr_0"]
                plt.figure()
                plt.hist(si[:,0],bins=50)

filenames = glob.glob("./pair_experiments/oplsaa*.npz")
for filename in filenames:
    si = np.load(filename)["arr_0"]
    plt.figure()
    plt.title(filename)
    plt.hist(si[:,0],bins=50)
