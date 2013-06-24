import ALA3
import numpy as np
import experiment_loader

for ff in ALA3.ff_list:
    predictions, measurements, uncertainties = experiment_loader.load(ff)
    phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)

    correlation = 50.
    n = 1.0 * len(ass_raw) / correlation
    raw_pops = 1.0 * np.bincount(ass_raw)
    raw_pops /= raw_pops.sum()
    raw_errs = np.sqrt(raw_pops*(1-raw_pops) / n)

    D0 = (predictions.mean() - measurements) / uncertainties
    raw_rms = (D0 ** 2).mean() ** 0.5

    out_dir = ALA3.data_directory + "/raw/"
    np.savetxt(out_dir + "raw_populations_%s.dat" % ff,raw_pops)
    np.savetxt(out_dir + "raw_uncertainties_%s.dat" % ff,raw_errs)
    np.savetxt(out_dir + "raw_rms_%s.dat" % ff, [raw_rms])
