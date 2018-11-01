import scipy.optimize as spo
import numpy as np
import torch

def lbfgsb(var, min_val, max_val, loss_fn, zero_grad_fn, **kwargs):
    x = var.detach().numpy()
    shape = x.shape
    x = x.ravel().astype(np.float64) # l-bfgs-b uses double apparently
    bounds = [(min_val, max_val)] * x.shape[0]
    def f(x):
        var.detach()[:] = torch.from_numpy(x.reshape(shape))
        loss = loss_fn(**kwargs)
        zero_grad_fn(**kwargs)
        loss.backward()
        f = loss.detach().numpy().astype(np.float64)
        g = var.grad.detach().numpy().ravel().astype(np.float64)
        return f, g
    x, f, d = spo.fmin_l_bfgs_b(f, x, bounds = bounds)
    var.detach()[:] = torch.from_numpy(x.reshape(shape))