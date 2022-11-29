# Levenberg-Marquardt algorithm
import torch
import numpy as np
from time import time

__all__ = ["LM"]

class LM(object):
    """
    based heavily on:
    @article{gavin2019levenberg,
        title={The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems},
        author={Gavin, Henri P},
        journal={Department of Civil and Environmental Engineering, Duke University},
        volume={19},
        year={2019}
    }
    """
    
    def __init__(self, model, lambda0 = None, max_iter = 100, epsilon4 = 1e-1, Lup = 11., Ldn = 9., L0 = 1e0, run_fit = True):

        self.model = model
        self.lambda_history = []
        self.lambda_new = lambda0
        self.lambda_old = lambda0
        self.max_iter = max_iter
        self.epsilon4 = epsilon4
        self.Lup = Lup
        self.Ldn = Ldn
        self.L = L0
        self.loss_history = []
        self.L_history = []
        self.decision_history = []
        self.iteration = 0
        self.Y = model.target.data.view(-1)
        self.W = 1 / model.target.variance.view(-1) if model.target.has_variance else None
        self.ndf = (len(self.Y) - len(self.lambda_new) + 1)
        self.J = None

        if run_fit:
            self.main_loop()


    def step(self, current_state):
        pass
    
    def main_loop(self):
        loss = 1e8
        h = torch.zeros(len(self.lambda_new))
        count_reject = 0
        count_finish = 0
        Ynew = torch.zeros(len(self.Y))
        while self.iteration < self.max_iter and count_reject < 12 and count_finish < 3 and torch.all(torch.isfinite(self.lambda_new)):
            print("---------iter---------")
            if self.iteration > 0:
                h = self.update_h()
                    
            with torch.no_grad():
                Ytmp = torch.clone(Ynew)
                Ynew = self.model.forward(self.lambda_new + h).view(-1)
                loss = torch.sum(((self.Y - Ynew)**2 if self.W is None else ((self.Y - Ynew)**2 * self.W))) / self.ndf
            if self.iteration == 0:
                self.decision_history.append("init")
            self.loss_history.append(loss.detach().cpu().item())
            self.L_history.append(self.L)
            self.lambda_history.append(np.copy((self.lambda_new + h).detach().cpu().numpy()))
            if self.iteration > 0:
                print("LM loss: ", loss, np.min(self.loss_history[:-1]), h)
                if 0 < (np.min(self.loss_history[:-1]) - loss) < 1e-6:
                    count_finish += 1
                else:
                    count_finish = 0
                rho = self.rho(np.min(self.loss_history[:-1]), loss, h) 
                print(rho)
                if rho > self.epsilon4:
                    print("accept")
                    self.decision_history.append("accept")
                    Yold = torch.clone(Ytmp)
                    self.lambda_old = self.lambda_new
                    self.lambda_new += h
                    self.L = max(1e-9, self.L / self.Ldn)
                    count_reject = 0
                elif count_reject < 4:
                    print("reject")
                    self.decision_history.append("reject")
                    self.L = min(1e9, self.L * self.Lup)
                    count_reject += 1
                    continue
                else:
                    print("reject")
                    self.decision_history.append("reject")
                    self.L = min(1e9, self.L * self.Lup)
                    count_reject += 1                    
            else:
                pass

            jac_time = time()
            if self.J is None or self.iteration < 2 or count_reject >= 4:
                self.update_J_AD(h)
            else:
                self.update_J_Broyden(h, Yold, Ynew)
            print("jac time: ", time() - jac_time)

            with torch.no_grad():
                self.update_hess()
                self.update_grad(Ynew)
            self.iteration += 1

    def update_h(self):
        count_reject = 0
        h = torch.zeros(len(self.grad))
        while count_reject < 4:
            try:
                with torch.no_grad():
                    h = torch.linalg.solve(self.hess + self.L*torch.abs(torch.diag(self.hess))*torch.eye(len(self.grad)), self.grad)
                break
            except Exception as e:
                print("reject err: ", e)
                self.L = min(1e9, self.L * self.Lup)
                count_reject += 1
        return h
    
    def update_J_AD(self, h):
        self.J = self.model.jacobian(self.lambda_new + h).view(-1,len(self.lambda_new))
        
        
    def update_J_Broyden(self, h, Yp, Yph):
        with torch.no_grad():
            self.J += torch.outer(Yph - Yp - torch.matmul(self.J, h),h) / torch.linalg.norm(h)

    def update_hess(self):
        if self.W is None:
            self.hess = torch.matmul(self.J.T, self.J)
        else:
            self.hess = torch.matmul(self.J.T, self.W.view(len(self.W),-1)*self.J)
            
    def update_grad(self, Yph):
        if self.W is None:
            self.grad = torch.matmul(self.J.T, (self.Y - Yph))
        else:
            self.grad = torch.matmul(self.J.T, self.W * (self.Y - Yph))
            
    def rho(self, Xp, Xph, h):
        return self.ndf*(Xp - Xph) / abs(torch.dot(h, self.L * (torch.abs(torch.diag(self.hess)) * h) + self.grad))

    def res(self):
        return self.lambda_history[np.argmin(self.loss_history)]
