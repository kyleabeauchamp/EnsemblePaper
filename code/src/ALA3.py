#model = "combined_shifts"
model = "maxent"

ff_list = ["amber96","amber99","amber99sbnmr-ildn","charmm27","oplsaa"]
prior_list = ["maxent", "MVN"]

num_samples = 5000000
thin = 100
burn = 5000
kfold = 2

stride = 1
cross_val_stride = 20

regularization_strength_dict = {
"amber96":5,
"amber99":1,
"amber99sbnmr-ildn":7,
"charmm27":6,
"oplsaa":4
}

data_dir = "/home/kyleb/dat/ala_lvbp/"
outdir = "/home/kyleb/src/kyleabeauchamp/EnsemblePaper/paper/figures/"

cross_val_filename = outdir + "../../data/cross_val.dat"
