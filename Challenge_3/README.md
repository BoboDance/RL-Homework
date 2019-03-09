# Challenge 3

Authors: Johannes Czech, Jannis Weil, Fabian Otto

## Vanilla Policy Gradient

In this section, we describe our experience with implementing vanilla policy gradients.

Our main focus for the implementation is the environment `Levitation-v1` from the [Quanser platform](https://git.ias.informatik.tu-darmstadt.de/quanser/clients).

### Implementation and observations

The implementation can be found in the python module `Challenge_3.REINFORCE`.

### Issues

The levitation environment appeared very trivial to us at first sight, but this appear not to be the case.
The agent has a tendency to get stuck between -810 and -850 total reward.
This holds true for many different learning algorithms such as PG, NPG, PPO with baseline parameter settings.
Random actions give a similar reward with very little variance.
That's why expect that there is a strong local optima.
Moreover, since there's no rendering option it was hard to tell what the agent is learning or if he's learning something 
at all.

### Results

By using a discrete action space with only two bins and small neural network with 8 hidden units, it was able to overcome
this optimum. However, this came at the cost of prematurely ending the episode.
```
Eval (198 episodes): -38.8739 +/- 2.7076 (253.5606 +/- 16.5262 steps)
```

## Natural Policy Gradient

In this section, we describe our experience with implementing NPG.

NPG was tested out on the `BallBalancerSim-v0` environment form the [Quanser platform](https://git.ias.informatik.tu-darmstadt.de/quanser/clients). Additionally, we created a model for the classic `Pendulum-v0` [environment](https://gym.openai.com/envs/Pendulum-v0/).

### Implementation and observations

The implementation can be found in the python module `Challenge_3.NPG`.

### Issues

### Results

## Natural Evolution Strategies

In this section, we describe our experience with implementing NES.

NES was tested out on the `BallBalancerSim-v0` environment form the [Quanser platform](https://git.ias.informatik.tu-darmstadt.de/quanser/clients). Additionally, we created a model for the classic `Pendulum-v0` [environment](https://gym.openai.com/envs/Pendulum-v0/).

### Implementation and observations

The implementation can be found in the python module `Challenge_3.NES`.


### Issues

### Results