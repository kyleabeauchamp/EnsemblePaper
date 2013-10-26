import numpy as np
from fitensemble import belt

belt.ne.set_num_threads(1)
kfold = 2
thin = 10
n_samples = 200000

measurements = np.array([0.0])
uncertainties = np.array([1.0])

n_predictions_list = np.array([100, 1000, 10000, 100000])
regularization_strength_list = np.array([0.75, 1.0, 1.5, 2.0, 2.5])

results = np.zeros((len(n_predictions_list), len(regularization_strength_list), 2))

for i, n_predictions in enumerate(n_predictions_list):
    predictions = np.random.normal(1.0, 1.0, size=(n_predictions, 1))
    bootstrap_index_list = np.array_split(np.arange(len(predictions)), kfold)
    for j, regularization_strength in enumerate(regularization_strength_list):
        model_factory = lambda predictions, measurements, uncertainties: belt.MaxEntBELT(predictions, measurements, uncertainties, regularization_strength)        
        train_chi, test_chi = belt.cross_validated_mcmc(predictions, measurements, uncertainties, model_factory, bootstrap_index_list, n_samples, thin=thin)
        print regularization_strength, train_chi.mean(), test_chi.mean()
        results[i,j] = train_chi.mean(), test_chi.mean()
