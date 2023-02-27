from .base import *
from .lm import *
from .gradient import *
from .langevin import *
from .iterative import *
"""
base: This module defines the base class AutoprofBase, 
      which is used as the parent class for all profiling algorithms in AutoProf. 
      This module contains helper functions used across multiple profiling algorithms, 
      such as computing gradients and making copies of models.

lm: This module defines the class LMProfiler, 
    which uses the Levenberg-Marquardt algorithm to perform profiling. 
    This algorithm adjusts the learning rate at each step to find the optimal value.

gradient: This module defines the class GradientProfiler, 
          which uses a simple gradient descent algorithm to perform profiling. 
          This algorithm adjusts the learning rate at each step to find the optimal value.

langevin: This module defines the class Langevin, 
          which uses the Langevin dynamics algorithm to perform profiling. 
          This algorithm simulates a physical system to find the optimal value.

iterative: This module defines the class iter, 
            which uses an iterative algorithm to perform profiling. 
            This algorithm repeatedly trains the model with different learning rates until it finds the optimal value.

"""