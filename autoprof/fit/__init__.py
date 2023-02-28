from .base import *
from .lm import *
from .gradient import *
from .langevin import *
from .iterative import *
"""
base: This module defines the base class BaseOptimizer, 
      which is used as the parent class for all algorithms in AutoProf. 
      This module contains helper functions used across multiple algorithms, 
      such as computing gradients and making copies of models.

LM: This module defines the class LMProfiler, 
    which uses the Levenberg-Marquardt algorithm. 
    This algorithm adjusts the learning rate at each step to find the optimal value.

Grad: This module defines the class GradientProfiler, 
          which uses a simple gradient descent algorithm. 
          This algorithm adjusts the learning rate at each step to find the optimal value.

langevin: This module defines the class MALA, 
          which uses the Langevin dynamics algorithm. 
          This algorithm simulates a physical system to find the optimal value.

Iterative: This module defines the class Iter, 
            which uses an iterative algorithm to perform profiling. 
            This algorithm repeatedly trains the model with different learning rates until it finds the optimal value.

"""