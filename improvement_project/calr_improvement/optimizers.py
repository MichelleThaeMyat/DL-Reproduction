import math
import torch
from torch.optim import Optimizer

def cosine_lr(epoch,epochs,initial=.01,minimum=.001):
    if epochs<2 or not 0<=epoch<epochs:raise ValueError('Invalid epoch')
    return minimum+.5*(initial-minimum)*(1+math.cos(math.pi*epoch/(epochs-1)))

class SmoothedCALR(Optimizer):
    """Bias-corrected EMA of absolute gradients replaces instantaneous magnitude."""
    def __init__(self,params,lr=.01,alpha=.1,rho=.9,eps=1e-8):
        if lr<=0 or alpha<=0 or eps<=0 or not 0<=rho<1:raise ValueError('Invalid parameters')
        super().__init__(params,dict(lr=lr,alpha=alpha,rho=rho,eps=eps))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                g=p.grad
                if g.is_sparse:raise RuntimeError('Sparse gradients unsupported')
                st=self.state[p]
                if not st:st.update(t=0,m=torch.zeros_like(p),q=torch.zeros_like(p))
                st['t']+=1;st['m'].mul_(.9).add_(g,alpha=1-.9)
                st['q'].mul_(group['rho']).add_(g.abs(),alpha=1-group['rho'])
                m=st['m']/(1-.9**st['t']);q=st['q']/(1-group['rho']**st['t'])
                p.add_(-group['lr']/((q+group['alpha']).sqrt()+group['eps'])*m)
        return loss
