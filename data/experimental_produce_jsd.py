import sys
sys.path.append("../src/") 

import pandas as pd


from plot_utils import  *

def save_df_to_dsk(data_df, name:str):
    with open(name+'.csv', 'a') as f:
        data_df.to_csv(f)

train_set = pd.read_csv('experimental_dir/Soil_L2.csv', header=None) .values
test_set = train_set # pd.read_csv('k1_dir/k1_test_set.csv', header=None) .values

gan_path = 'L2_experimental'
jsd_table = build_table2(gan_path, test_set)


save_df_to_dsk(jsd_table, "tabla_experimental_L2")
