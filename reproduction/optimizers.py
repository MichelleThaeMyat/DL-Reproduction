"""New implementations targeting the released repository's numerical behavior.

Source of method and reference behavior: Gayan225/CALR-Optimizer (MIT).
These are repository-compatible algorithms, not a claim of paper-equation fidelity.
"""
import torch
from torch.optim import Optimizer


class GradientCALR(Optimizer):
    """Momentum divided by sqrt(abs(current gradient) + alpha).

    exploratory=True preserves the separate toy scripts' floating-point operation
    order. Their literal 0.1 differs slightly from the main script's 1 - 0.9.
    Unused second moments are deliberately omitted.
    """
    def __init__(self, params, lr=0.01, alpha=0.1, eps=1e-8, exploratory=False):
        if lr <= 0 or alpha <= 0 or eps <= 0:
            raise ValueError("lr, alpha and eps must be positive")
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps,
                                     exploratory=exploratory))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported")
                state = self.state[parameter]
                if not state:
                    state.update(iteration=0, momentum=torch.zeros_like(parameter))
                state['iteration'] += 1
                if group['exploratory']:
                    state['momentum'] = 0.9 * state['momentum'] + 0.1 * gradient
                else:
                    state['momentum'].mul_(0.9).add_(gradient, alpha=1 - 0.9)
                corrected = state['momentum'] / (1 - 0.9 ** state['iteration'])
                scale = group['lr'] / ((gradient.abs() + group['alpha']).sqrt() + group['eps'])
                if group['exploratory']:
                    parameter.sub_(scale * corrected)
                else:
                    parameter.add_(-scale * corrected)
        return loss


class HimmelblauCALR(Optimizer):
    """The repository's Adam-style synthetic variant, optionally clipped.

    alpha is absent because the reference accepts it but never uses it.
    cosine_steps reproduces the reference's cumulative schedule, NOT standard
    cosine annealing. Those variants are disabled in the original default run.
    """
    def __init__(self, params, lr=0.01, clip=None, cosine_steps=None):
        if lr <= 0 or (clip is not None and clip <= 0):
            raise ValueError("lr and clipping threshold must be positive")
        if cosine_steps is not None and cosine_steps <= 0:
            raise ValueError("cosine_steps must be positive")
        super().__init__(params, dict(lr=lr, clip=clip, cosine_steps=cosine_steps,
                                     schedule_step=0))

    @torch.no_grad()
    def step(self, closure=None):
        import numpy as np
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group['cosine_steps'] is not None:
                phase = np.pi * group['schedule_step'] / group['cosine_steps']
                group['lr'] *= 0.5 * (1 + np.cos(phase))
                group['schedule_step'] += 1
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if group['clip'] is not None:
                    gradient = gradient.clamp(-group['clip'], group['clip'])
                state = self.state[parameter]
                if not state:
                    state.update(iteration=0, first=torch.zeros_like(parameter),
                                 second=torch.zeros_like(parameter))
                state['iteration'] += 1
                state['first'].mul_(0.9).add_(gradient, alpha=1 - 0.9)
                state['second'].mul_(0.999).addcmul_(gradient, gradient, value=1 - 0.999)
                numerator = state['first'] / (1 - 0.9 ** state['iteration'])
                denominator = (state['second'] / (1 - 0.999 ** state['iteration'])).sqrt() + 1e-8
                parameter.add_(-group['lr'] / denominator * numerator)
        return loss


class DiagonalHessianCALR:
    """Small-task reference-compatible diagonal Hessian experiment.

    The reference updates one parameter tensor before computing the next tensor's
    Hessian. Preserve that unusual order for comparison. This is NOT a corrected
    simultaneous Hessian method and is inappropriate as a production optimizer.
    """
    def __init__(self, model, objective, inputs, targets, lr=0.01, damping=0.1):
        self.model, self.objective = model, objective
        self.inputs, self.targets = inputs, targets
        self.lr, self.damping = lr, damping

    def step(self):
        self.model.zero_grad(set_to_none=True)
        value = self.objective(self.model(self.inputs), self.targets)
        value.backward(create_graph=True)
        for parameter in self.model.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            diagonal = torch.zeros_like(parameter)
            for index in range(parameter.numel()):
                component = gradient.reshape(-1)[index]
                if component.requires_grad:
                    derivative = torch.autograd.grad(component, parameter,
                                                     retain_graph=True, allow_unused=True)[0]
                    if derivative is not None:
                        diagonal.reshape(-1)[index] = derivative.reshape(-1)[index]
            # .data is intentional ONLY to reproduce the reference's in-graph
            # sequential mutation; changing it would change the experiment.
            parameter.data.sub_(self.lr / (diagonal.abs() + self.damping + 1e-8) * gradient)
        self.model.zero_grad(set_to_none=True)  # break backward(create_graph) cycles
        return value.detach()
