# MicrobiotaGAN
This repo contains code that supports the undergrad thesis that I wrote. 
The problem we tackle is data augmentation for soil microbiome samples. Among different generative models, a GAN is used. In particular, a Wasssertein GAN with gradient penalty.

This repo contains code for:
+ Automating experiments
+ Custom visualizations
+ Metrics to assess the quality of samples. _Usually, images would use an inception score, but our data is compositional_.


## Attributions/Thanks

This project borrowed some code for the [WGAN-GP](https://arxiv.org/abs/1704.00028) implementation from @ChengBinJin.
