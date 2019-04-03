import numpy as np

dim = 100
mean_vec = np.zeros(dim)

std = 0.1
cov_matrix = np.multiply(std**2, np.identity(dim))

# print(mean_vec)
# print(cov_matrix)   
n_samples = 10000

print("====> Creating samples <====")
x_train =[ np.random.multivariate_normal(mean_vec, cov_matrix, ()) for k in range(n_samples) ]
print("====> Samples have been created <====")

"""
import pickle

with open('train_set.pkl', 'wb') as f:
  pickle.dump(x_train, f)

"""
print("===> Writing data to ssd")
import pandas as pd

df = pd.DataFrame(x_train)
with open('my_data.csv', 'a') as f:
    df.to_csv(f, header=False, index=False)
