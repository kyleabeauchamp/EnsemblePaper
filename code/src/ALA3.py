#model = "combined_shifts"
model = "maxent"

ff_list = ["amber96","amber99","amber99sbnmr-ildn","charmm27","oplsaa"]
prior_list = ["maxent", "MVN"]

train_keys = ['JC_2_J3_HN_Cprime', 'CS_2_CA', 'CS_2_H', 'JC_3_J2_N_CA', 'CS_2_CB', 'JC_2_J3_HN_CB']
test_keys = ["JC_2_J3_HN_HA" , "JC_2_J3_HA_Cprime", "JC_2_J1_N_CA"]
all_keys = []
all_keys.extend(train_keys)
all_keys.extend(test_keys)

bw_num_samples = 1000000
num_samples = 5000000
thin = 100
burn = 5000
kfold = 2

stride = 1
cross_val_stride = 20

regularization_strength_dict = {"maxent":
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


data_dir = "/home/kyleb/dat/ala_lvbp/"
outdir = "/home/kyleb/src/kyleabeauchamp/EnsemblePaper/paper/figures/"

cross_val_filename = outdir + "../../data/cross_val.dat"
