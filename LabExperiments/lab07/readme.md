Experiment 7

Aim: This experiment is to understand the implementation of Vision Transformers and its utility in classification applications.

Data Files: CIFAR Dataset

1. Create a class “patchembedding” which creates patches from given images, flattens it and adds positional embedding. The resultant is given as input to transformer architecture. Create a class “vit” which describes the architecture of the transformer. Print the layer details (using torchinfo) of the architecture.
2. Load CIFAR dataset using Pytorch API. Create a training and evaluation loop for the architecture developed in step 1. Print the training and evaluation results.
