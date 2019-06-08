import sys
sys.path.append("../src/") 

import pandas as pd


from plot_utils import  *

def save_df_to_dsk(data_df, name:str):
    with open(name+'.csv', 'a') as f:
        data_df.to_csv(f)

train_set = pd.read_csv('k4_dir/k4_train_set.csv', header=None) .values
test_set = pd.read_csv('k4_dir/k4_test_set.csv', header=None) .values

gan_path = 'redo_k4/'
jsd_table = build_table2(gan_path, test_set)


save_df_to_dsk(jsd_table, "low_lr_k4_2000_epochs")
