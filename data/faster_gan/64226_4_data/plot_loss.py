import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

epochs = pd.read_csv('_epoch_record.csv', header=None).values
gen_loss = pd.read_csv('_gen_loss.csv', header=None).values
disc_loss = pd.read_csv('_disc_loss.csv', header=None).values

n_samples  = str(pd.read_csv('training_indices.csv', header=None).shape[0])
tag = '('+n_samples+' samples)'

# generator

plt.figure(figsize=(18.5, 10.5))
plt.plot(epochs,gen_loss)
plt.title("Generator Loss "+tag)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.savefig('generator_loss.png', dpi=300)
plt.close()


# discriminator
plt.figure(figsize=(18.5, 10.5))

plt.plot(epochs,disc_loss)
plt.title("Discriminator Loss "+ tag)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.savefig('discriminator_loss.png', dpi=300)
plt.close()

# both
plt.figure(figsize=(18.5, 10.5))

plt.plot(epochs,disc_loss, label='Discriminator')
plt.plot(epochs,gen_loss, label='Generator')
plt.legend()

plt.title("GAN Loss " +tag)
plt.xlabel("Epochs")
plt.ylabel("Loss")
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

plt.savefig('gan_loss.png', dpi=300)

plt.close()