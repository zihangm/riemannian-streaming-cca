# riemannian-streaming-cca

This repository contains the reference code for our paper [An Online Riemannian PCA for Stochastic Canonical Correlation Analysis](https://arxiv.org/pdf/2106.07479.pdf) (Neurips-2021)

## Requirements
For CCA experiments on MNIST, CIFAR10 and Mediamill:
* Matlab
* [manopt](https://www.manopt.org/) for Matlab

For DeepCCA experiments:
* Python 3
* Pytorch 1.5+

## CCA experiments
Download the mat files for MNIST, CIFAR10 and Mediamill from google drive [here](https://drive.google.com/file/d/1jUnhhXfepxUQtxJJrMe_FKNu08ljuxAs/view?usp=sharing) and put them in ./CCA

Use Matlab to run demo_run.m

## DeepCCA experiments
To be added soon

## Reference
If you find our work useful, please consider citing our paper.
```
@article{meng2021online,
  title={An Online Riemannian PCA for Stochastic Canonical Correlation Analysis},
  author={Meng, Zihang and Chakraborty, Rudrasis and Singh, Vikas},
  journal={arXiv preprint arXiv:2106.07479},
  year={2021}
}
```


