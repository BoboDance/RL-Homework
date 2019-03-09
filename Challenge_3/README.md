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

We first tried to implement NPG as a simple extension of REINFORCE by sampling the fisher information matrix from
the policy gradient, 

<img src="https://raw.githubusercontent.com/BoboDance/RL-Homework/master/Challenge_3/supplementary/fisher_information.png" height="70"></img>

but we soon realized that this is too computationally inefficient.

We then build upon the [solution by Woongwon Lee et al.](https://github.com/reinforcement-learning-kr/pg_travel/), where
the natural gradient step is performed by calculating the fisher information matrix using the second derivative of the KL-divergence
(see [this blog post from Boris](http://www.boris-belousov.net/2016/10/16/fisher-vs-KL/)) and then using conjugate gradient
instead of explicitly creating the inverse fisher information matrix.

This way, we are able to achieve good results. 

### Issues

As the plain npg implementation uses a fixed alpha for the parameter update,
we tried to improve it by using the "normalized" step size as menioned in the
[paper from Rajeswaran et al.](https://arxiv.org/pdf/1703.02660.pdf): 

<img src="https://raw.githubusercontent.com/BoboDance/RL-Homework/master/Challenge_3/supplementary/normalized_step_size.png" height="70">

but unfortunately, we were not able to improve our policy in comparison to the plain update step.

Additionally, we had some problems when predicting the mean and the std of the policies normal distribution together with
our policy network. Using an independent network parameter instead helped a lot and caused improved stability during training.

### Results

We see stable training behaviour like the following on `BallBalancerSim-v0` very frequently:

<img src="https://raw.githubusercontent.com/BoboDance/RL-Homework/master/Challenge_3/supplementary/npg_episode_reward.png" height="200"></img>

<img src="https://raw.githubusercontent.com/BoboDance/RL-Homework/master/Challenge_3/supplementary/npg_episode_steps.png" height="200"></img>

One can see that that the number of steps converge to the maximum episode steps (1000) and the reward increases until the maximum step size
is reached. The illustrated policy receives an average reward of 365.0112 +/- 117.8526 (22 evaluation episodes).
Corresponding parameters can be found in `NPG/natural_test.py` (use `step_size=0.3` to reproduce the graphs).

Lowering the step size also lowers the variance in the policy, we get 403.4890 +/- 63.5127 (40 evaluation episodes) for 
`step_size=0.2` after training 1000 episodes.

## Natural Evolution Strategies

In this section, we describe our experience with implementing NES.

NES was tested out on the `BallBalancerSim-v0` environment form the [Quanser platform](https://git.ias.informatik.tu-darmstadt.de/quanser/clients). Additionally, we created a model for the classic `Pendulum-v0` [environment](https://gym.openai.com/envs/Pendulum-v0/).

### Implementation and observations

The implementation can be found in the python module `Challenge_3.NES`.


### Issues

### Results