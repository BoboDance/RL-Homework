# Challenge 2

Authors: Jannis Weil, Johannes Czech, Fabian Otto

## Least-Squares Policy Iteration (LSPI)

In this section, we describe our experience with implementing LSPI based on the paper from [Lagoudakis & Parr](http://jmlr.csail.mit.edu/papers/v4/lagoudakis03a.html).

Our main focus for the implementation is the environment `CartpoleStabShort-v0` from the [Quanser platform](https://git.ias.informatik.tu-darmstadt.de/quanser/clients).

### Implementation and observations

The implementation can be found in the python module `Challenge_2.LSPI`.

#### Discretization of actions
For the discretization of actions we decided to use [-5, 0, +5], this allows the cartpole to maintain its upright postition by choosing action [0]. Removing [0] from the possible actions highly reduced the performance for us. We assume this happends due to the need to select going to the left and right, which are both equaly good/bad at the fully upright position, i.e. the algorithm is harming its own performance as both action lead to worst states.

#### Feature functions
Finding a good feature function is the key challenge of LSPI. The algorithm itself does not require a lot of complex computation, the main part can be found in the LSTDQ-Model (see below). Therefor, finding a good feature function decides over success and failure. We implemented RBF features as well as Fourier Features (both can be found in`Challenge_2.LSPI.BasisFunctions`).
During our tests, the Fourier features worked significantly better and we were not able to learn a consitent policy with RBF features. 
One reason for this is, in our opinion, the large hyperparameter space for RBFs. It is necessary to tune the RBF centers as well as the length scales.

![Bild](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3b2a39c3806996aaaf57bbe99bd67de0e2c529af/19-Figure7-1.png)

#### Other
* Normalization helped a lot (often it did not converge otherwise)
* "Optimized" LSTDQ is slower? We need manual loops there.
* Gets more stable when using more samples: Add example comparions with 25k, 50k and 100k samples???
* 

### Results

* Very good stab policy (20k) after few iterations, delta converges to zero.
  [GIF]

## Deep Q-Learning (DQN)

In this section, we describe our experience with implementing DQN based on [Deepmind's paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

DQN was tested out on the `CartpoleSwingShort-v0` environment form the [Quanser platform](https://git.ias.informatik.tu-darmstadt.de/quanser/clients). Additionally, we created a model for the classic `Pendulum-v0` [environment](https://gym.openai.com/envs/Pendulum-v0/).

### Implementation and observations

The implementation can be found in the python module `Challenge_2.DQN`.

* Decreasing epsilon for exploration
* Model type (experiments with different architectures)
* More steps before updating target Q for better stability
* Lower actions for stability (e.g. [-5, +5] instead of [-24, +24] )
* Is easier with reward shaping (e.g. punishing when the agent comes close to the border)

### Results

Learning
[Plot of loss curve]

[Plot of average reward development]

<!-- Haben wir eine policy für den Stab task???? 
    Jannis: Nicht dass ich wüsste, das hatte ich nur irgendwann mal ausprobiert und es hatte damals ok geklappt. Aber war lange nicht so gut wie unser LSPI.
-->

* Pendulum-v0: DQN learns a very good policy quite fast
  [GIF] 

* Swingup: Propeller policy
  [GIF]
