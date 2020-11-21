This code is tested with Pytorch 0.4 and Chainer 5.0, used for the CIFAR generation experiment.

Run CIFAR_NRF_unsupervised.py to perform unsupervised training with NRF.

Run CIFAR_NRF_supervised.py to perform supervised training with NRF (still unconditional generation).

To compute inception score and FID for generated images, first download inception model:

```
python evaluations/chainer/download.py --outfile=../data/inception_model
```

To compute inception score and FID, there are two versions, Tensorflow and Chainer version in evaluations/tf and evaluations/chainer folder separately, and the Chainer version is used to fairly compare with Spectral normalization GAN (SNGAN).

The stat files used to compute FID (the statistics of 10,000 CIFAR test images) are in ../data folder.

