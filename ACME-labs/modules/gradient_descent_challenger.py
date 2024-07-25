import scipy.optimize as opt
from autograd import numpy as np
from autograd import grad


def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #iterate on gradient descent
    for iters in range(1,maxiter+1):
        Dfx0 = Df(x0)
        
        #calculate ak, x1
        ak = opt.minimize_scalar(lambda a: f(x0-a*Dfx0)).x
        x1 = x0-ak*Dfx0
        
        #stopping condition
        if np.linalg.norm(Df(x1))<tol:
            return x1, True, iters
        x0=x1
    
    return x1, False, iters


class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        #compute L and DL
        L = lambda b: np.sum([np.log(1+np.exp(-1*(b[0]+b[1]*x))) + (1-y)*(b[0]+b[1]*x)])
        DL = grad(L)
        
        #compute and store b
        self.b = steepest_descent(L, DL, guess)[0]

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        #compute and return sigma(x)
        return 1/(1+np.exp(-1*(self.b[0]+self.b[1]*x)))
