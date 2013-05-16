model = "combined_shifts"

ff_list = ["amber96","amber99","amber99sbnmr-ildn","charmm27","oplsaa"]

num_samples = 25000
thin = 100
burn = 5000
kfold = 2

stride = 1
cross_val_stride = 10

regularization_strength_dict = {
"amber96":5,
"amber99":5,
"amber99sbnmr-ildn":7,
"charmm27":6,
"oplsaa":4
}

data_dir = "/home/kyleb/dat/ala_lvbp/"
outdir = "/home/kyleb/src/kyleabeauchamp/EnsemblePaper/paper/figures/"