import experiment_loader
import ALA3
import numpy as np
from fitensemble import belt
import sys

belt.ne.set_num_threads(1)
ff = "amber96"
regularization_strength = 1E-5
stride = 40


predictions, measurements, uncertainties = experiment_loader.load(ff, stride=stride)
model_factory = lambda predictions, measurements, uncertainties: belt.MaxEntBELT(predictions, measurements, uncertainties, regularization_strength, log_det_correction=True)

bootstrap_index_list = np.array_split(np.arange(len(predictions)), ALA3.kfold)
train_chi, test_chi = belt.cross_validated_mcmc(predictions.values, measurements.values, uncertainties.values, model_factory, bootstrap_index_list, ALA3.num_samples, thin=ALA3.thin)

print regularization_strength, train_chi.mean(), test_chi.mean()
print("%s,%f,%d,%d,%f,%f \n"% (ff, regularization_strength, ALA3.cross_val_stride, ALA3.num_samples, train_chi.mean(), test_chi.mean()))


"""
OLD, 1.0 * lambda * logdet(H)
amber96,0.040000,20,10000000,11.319633,11.321108
amber96,0.030000,20,10000000,10.000474,10.002316
amber96,0.020000,20,10000000,8.063870,8.067065 
amber96,0.010000,20,10000000,5.129576,5.139750
"""

"""
New: 0.5 * logdet(H)

amber96,0.001000,20,10000000,17.171235,17.171307
amber96,0.005000,20,10000000,17.169116,17.168968 
amber96,0.010000,20,10000000,17.167470,17.167675
amber96,0.050000,20,10000000,17.175696,17.175888

amber96,0.000010,20,10000000,17.173472,17.173432
