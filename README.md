# Reverse engineering recurrent neural networks with Jacobian switching linear dynamical systems

This repository is the official implementation of Reverse engineering recurrent neural networks with Jacobian switching linear dynamical systems (https://arxiv.org/abs/2111.01256). 

## Requirements
All of the requirements are already installed in the Google Colab environment (https://colab.research.google.com/) in the Colab notebooks provided. This is the recommended environment to use these notebooks.

To run the code on your own machine you will need to install JAX (https://github.com/google/jax#installation).

## Training and Evaluation

Instructions for how to train and evaluate the models for the 3-bit memory task and the context-dependent integration task are included in the Colab notebooks : JSLDS_3bit_memory_notebook.ipynb and JSLDS_context_integration_notebook.ipynb.

## Pre-trained Models

We include pre-trained weights for both the co-trained models as well as the standard models without JSLDS co-training for both the 3-bit memory task and the context-dependent integration task.
