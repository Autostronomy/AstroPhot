from .base import *
from .lm import *
from .gradient import *
from .iterative import *
from .hmc import *
from .mhmcmc import *
from .nuts import *

"""
base: This module defines the base class BaseOptimizer, 
      which is used as the parent class for all optimization algorithms in AutoProf. 
      This module contains helper functions used across multiple optimization algorithms, 
      such as computing gradients and making copies of models.

LM: This module defines the class LM, 
    which uses the Levenberg-Marquardt algorithm to perform optimization. 
    This algorithm adjusts the learning rate at each step to find the optimal value.

Grad: This module defines the class Gradient-Optimizer, 
      which uses a simple gradient descent algorithm to perform optimization. 
      This algorithm adjusts the learning rate at each step to find the optimal value.

Iterative: This module defines the class Iter, 
            which uses an iterative algorithm to perform Optimization. 
            This algorithm repeatedly fits each model individually until they all converge.

"""
