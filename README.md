Python Optimiser API
This project provides a flexible and extensible optimiser API in Python, designed for use in machine learning tasks. It is built to be framework-agnostic, working seamlessly with both NumPy arrays and PyTorch Tensors.

Key Features
Extensible: Easily add new optimisers by inheriting from the OptimiserParentClass and registering them with a decorator.

Framework-Agnostic: Designed to work with raw NumPy parameters or PyTorch models.

Standard Interface: Provides a familiar API with step() and zero_grad() methods.

Utility Functions: Includes built-in support for features like weight decay.

Available Optimisers
The following optimisers are currently implemented:

SGD: Standard Stochastic Gradient Descent.

NAG: Nesterov Accelerated Gradient.

Adam: Adaptive Moment Estimation.

MUON: Momentum Updating Orthogonalisation Normalisation

This API allows you to easily experiment with different optimisation algorithms for your machine learning models. See the HOW_TO_USE.md for detailed instructions on getting started.

The next goal for this is to try and implement the orthogonalized gradient momentum using Newton-Schulz iteration (or Muon). A beautiful video can be seen explaining this on Jia-Bin Huang's Youtube channel 
https://www.youtube.com/watch?v=bO5nvE289ec&t=952s
or here on the Arxiv paper (Jingyuan Liu et al 2025)
https://arxiv.org/pdf/2502.16982
