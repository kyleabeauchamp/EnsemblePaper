import pandas as pd

ff_map = {  # Used to rename forcefields from 'amber96' to 'ff96' without having to re-name the raw data files.
"""#amber99""":"""#ff99""",  # Some messed up values of cross validation that were discarded
"amber96":"ff96",
"amber99":"ff99",
"amber99sbnmr-ildn":"ff99sbnmr-ildn",
"oplsaa":"oplsaa",
"charmm27":"charmm27"
}

#model = "combined_shifts"
model = "maxent"

ff_list = ["amber96","amber99","amber99sbnmr-ildn","charmm27","oplsaa"]
mapped_ff_list = [ff_map[x] for x in ff_list]
prior_list = ["maxent", "dirichlet", "MVN"]

#train_keys = ['JC_2_J3_HN_Cprime', 'CS_2_CA', 'CS_2_H', 'JC_3_J2_N_CA', 'CS_2_CB', 'JC_2_J3_HN_CB']
#test_keys = ["JC_2_J3_HN_HA" , "JC_2_J3_HA_Cprime", "JC_2_J1_N_CA"]

tuples = [("JC", 2, "J3_HN_Cprime"), ("JC", 3, "J2_N_CA"), ("JC", 2, "J3_HN_CB"), ("CS", 2, "CA"), ("CS", 2, "H"), ("CS", 2, "CB")]
train_keys = pd.MultiIndex.from_tuples(tuples, names=("experiment", "resid", "name"))

tuples = [("JC", 2, "J3_HN_HA"), ("JC", 2, "J3_HA_Cprime"), ("JC", 2, "J1_N_CA"), ("CS", 2, "HA")]
test_keys = pd.MultiIndex.from_tuples(tuples, names=("experiment", "resid", "name"))


#all_keys = []
#all_keys.extend(train_keys)
#all_keys.extend(test_keys)

bw_num_samples = 1000000
#num_samples = 5000000
num_samples = 10000000
thin = 100
burn = 5000
kfold = 2
num_blocks = 10

stride = 1
cross_val_stride = 20

regularization_strength_dict = {"maxent":
{
"amber96":10,
"amber99":4,
"amber99sbnmr-ildn":100,
"charmm27":6,
"oplsaa":15
}
,
"MVN":
{
"amber96":6,
"amber99":1,
"amber99sbnmr-ildn":100,
"charmm27":4,
"oplsaa":12
},
"dirichlet":
{
"amber96":7,
"amber99":1.2,
"amber99sbnmr-ildn":100,
"charmm27":4,
"oplsaa":13
}
}


old_regularization_strength_dict = {"maxent":
{
"amber96":5,
"amber99":1,
"amber99sbnmr-ildn":1,
"charmm27":1,
"oplsaa":7
}
,
"MVN":
{
"amber96":8,
"amber99":1,
"amber99sbnmr-ildn":2,
"charmm27":1,
"oplsaa":14
}
}


data_directory = "/home/kyleb/dat/ala_lvbp/"

outdir = "/home/kyleb/src/kyleabeauchamp/EnsemblePaper/paper/figures/"
cross_val_filename = outdir + "../../data/cross_val.dat"
chi2_filename = outdir + "../../data/chi2.dat"
experiment_filename = outdir + "../../data/experimental_data.csv"
