import experiment_loader
import ALA3
import numpy as np
from fitensemble import lvbp
lvbp.ne.set_num_threads(3)

ff = "amber99sbnmr-ildn"
prior = "maxent"
reg_list = [1.0, 2.0, 5.0]

directory = "%s/%s" % (ALA3.data_dir , ff)
out_dir = directory + "/cross_val/"

num_samples = 150000

#predictions, measurements, uncertainties = experiment_loader.load(directory, keys=[("CS", 2, "H"), ("CS", 2, "HA")])
predictions, measurements, uncertainties = experiment_loader.load(directory)

data = np.zeros((len(reg_list), 2))
for k, regularization_strength in enumerate(reg_list):
    if prior == "maxent":
        model_factory = lambda predictions, measurements, uncertainties: lvbp.MaxEnt_LVBP(predictions, measurements, uncertainties, regularization_strength)
    else:
        precision = np.cov(predictions.values.T)
        model_factory = lambda predictions, measurements, uncertainties: lvbp.MVN_LVBP(predictions, measurements, uncertainties, regularization_strength, precision=precision)
    bootstrap_index_list = np.array_split(np.arange(len(predictions)), ALA3.kfold)
    train_chi, test_chi = lvbp.cross_validated_mcmc(predictions.values, measurements.values, uncertainties.values, model_factory, bootstrap_index_list, num_samples, thin=ALA3.thin)
    data[k] = train_chi.mean(), test_chi.mean()

plot(reg_list, data)
