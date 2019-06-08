import sys
sys.path.append("../src/") 

import pandas as pd


from plot_utils import  *

def save_df_to_dsk(data_df, name:str):
    with open(name+'.csv', 'a') as f:
        data_df.to_csv(f)

train_set = pd.read_csv('k3_dir/k3_train_set.csv', header=None) .values
test_set = pd.read_csv('k3_dir/k3_test_set.csv', header=None) .values

gan_path = 'pre_improved_k3/'
jsd_table = build_table2(gan_path, test_set)


save_df_to_dsk(jsd_table, "k3_2000_epochs")
