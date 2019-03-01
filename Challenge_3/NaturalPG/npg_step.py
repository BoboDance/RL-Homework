import torch

import numpy as np

from Challenge_3.NaturalPG.ValueModel import train_model
from Challenge_3.Util import get_returns_torch

# The natural policy gradient step based on https://github.com/reinforcement-learning-kr/pg_travel


def fisher_vector_product(actor, states, p):
    p.detach()
    kl = kl_divergence(new_actor=actor, old_actor=actor, states=states)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)

    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + 0.1 * p


# from openai baseline code: https://github.com/openai/baselines/blob/master/baselines/common/cg.py
def conjugate_gradient(actor, states, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = fisher_vector_product(actor, states, p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def optimization_step(actor, memory, gamma, critic=None, optimizer_critic=None):
    states = np.vstack(memory[:, 0])
    rewards = memory[:, 2]
    # create a single tensor out of the log confidence
    log_confidence = torch.stack(list(memory[:, 4]))

    # Get returns
    returns = get_returns_torch(rewards, gamma, normalize=True)

    # Train critic to improve the confidence weighting (we basically get the advantage)
    if critic is not None:
        critic_loss = train_model(critic, states, returns, optimizer_critic)
        returns = returns - critic(torch.Tensor(states))
    else:
        critic_loss = 0

    # Get gradient of loss and hessian of kl
    actor_loss = (returns * log_confidence).mean()
    loss_grad = torch.autograd.grad(actor_loss, actor.parameters())
    loss_grad = flat_grad(loss_grad)
    step_dir = conjugate_gradient(actor, states, loss_grad.data, nsteps=10)

    # Get step direction and step size
    params = flat_params(actor)
    # shs = 0.5 * (step_dir * fisher_vector_product(actor, states, step_dir)).sum(0, keepdim=True)
    # step_size = 1 / torch.sqrt(shs / 0.01)[0]
    step_size = 0.3
    new_params = params + step_size * step_dir

    # Update the actor
    update_model(actor, new_params)

    return actor_loss, critic_loss


def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_divergence(new_actor, old_actor, states):
    mu, std = new_actor(torch.Tensor(states))
    logstd = torch.log(std)
    mu_old, std_old = old_actor(torch.Tensor(states))
    logstd_old = torch.log(std_old).detach()
    mu_old = mu_old.detach()
    std_old = std_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)