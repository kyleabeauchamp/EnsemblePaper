import ALA3
import numpy as np

regularization_strength_list = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

for ff in ALA3.ff_list:
    for prior in ALA3.prior_list:
        for regularization_strength in regularization_strength_list:
            try:
                directory = "%s/%s" % (ALA3.data_dir , ff)
                out_dir = directory + "/cross_val/"
                train_chi, test_chi = np.loadtxt(out_dir+"/%s-reg-%d-stride%d-score.dat" % (prior, regularization_strength, ALA3.cross_val_stride))
                print("%s %s %f %d %d %f %f"% (ff, prior, regularization_strength, ALA3.cross_val_stride, ALA3.num_samples, train_chi, test_chi))
            except:
                pass
