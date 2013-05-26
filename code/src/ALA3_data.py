import numpy as np
import pandas as pd

measurements = pd.Series({  # Numbering from Table S3 in Schwalbe
"JC_2_J3_HN_HA"        :5.68,  # highly correlated to other measurements, r ranges from 0.71 to 0.997
"JC_2_J3_HN_Cprime"    :1.13,
"JC_2_J3_HA_Cprime"    :1.84,  # highly correlated to other measurements.  r > 0.7
"JC_2_J3_HN_CB"        :2.39,
"JC_2_J1_N_CA"         :11.34,  #  highly correlated to other measurements.  r > 0.7
"JC_3_J2_N_CA"         :8.45,

"CS_2_H":8.571,
"CS_2_HA":4.355,
"CS_2_CA":52.38,
"CS_2_CB":19.21
})


measurements = {(
