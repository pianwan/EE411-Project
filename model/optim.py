import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable


class BetaLASSO(Optimizer):
    def __init__(self, params, beta, lr, lambda_):
        defaults = dict(lr=lr, beta=beta, lambda_=lambda_)
        super(BetaLASSO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BetaLASSO, self).__setstate__(state)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr, beta, lambda_ = group['lr'], group['beta'], group['lambda_']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p.data.add_(d_p + lambda_ * torch.sign(p.data), alpha=-lr)
                # Beta-LASSO thresholding
                p.data[
                    (p.data > -beta * lambda_) & (p.data < beta * lambda_)
                    ] = 0

        return loss
