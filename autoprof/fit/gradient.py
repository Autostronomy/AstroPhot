# Traditional gradient descent with Adam
import torch
import numpy as np
from time import time

__all__ = ["Grad"]

class Grad(object):

    def __init__(self, model, lambda0 = None, max_iter = None, method = "NAdam", run_fit = True):

        self.max_iter = 100*len(lambda0)

        if run_fit:
            self.main_loop()

    def step(self, current_state):
        pass
