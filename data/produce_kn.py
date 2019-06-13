import sys
sys.path.append("../src/") 

import pandas as pd


from plot_utils import  *

def save_df_to_dsk(data_df, name:str):
    with open(name+'.csv', 'a') as f:
        data_df.to_csv(f)
print("Give the K of training")
kn = input()
train_set = pd.read_csv(kn+'_dir/'+kn+'_train_set.csv', header=None) .values
test_set = pd.read_csv(kn+'_dir/'+kn+'_test_set.csv', header=None) .values

gan_path = 'redo_'+kn
jsd_table = build_table2(gan_path, test_set)


save_df_to_dsk(jsd_table, "low_lr_"+kn+"_2000_epochs")
