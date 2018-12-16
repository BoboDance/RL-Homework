# Challenge 1 - Dynamic Programming

Environments: 
 * Pendulum-v0
 * Qube-v0

## Learning the Dynamics and Rewards

We tested out different model types for learning the dynamics and rewards

### Random Forest

![dynamics state 0](./Plots/Pendulum/Random_Forrest/dynamics_state0.png)
![dynamics state 01](./Plots/Pendulum/Random_Forrest/dynamics_state1.png)
![rewards](./Plots/Pendulum/Random_Forrest/rewards.png)

### Gaussian Process

![dynamics state 0](./Plots/Pendulum/Gaussian_Process/dynamics_state0.png)
![dynamics state 01](./Plots/Pendulum/Gaussian_Process/dynamics_state1.png)
![rewards](./Plots/Pendulum/Gaussian_Process/rewards.png)

### Neural Network

Later on we switched to using a Multi Layer Perceptron with the following architecture:
* 3 hidden layers each having 200 units
* 2 Dropout Layers with 30% rate between last after 2nd and third hidden layer

We use RMSProp as the optimizer with learning rate of `1e-3` and train for 150 epochs.
The learning rate is reduced by factor of 2 every 50 epochs.
All available sample 10,000 sample for Pendulum-v0 are used.
Training both networks takes about 3 minutes.
After training the network weights are stored and can be reloaded via `load_model = False` in 
[challenge1.py](./challenge1.py#L36).

![dynamics state last](./Plots/Pendulum/NN/Pendulum-v0_Dynamics.png)
![dynamics state last](./Plots/Pendulum/NN/Pendulum-v0_Reward.png)


## Finding the right bin sizes
One important detail is that two bins are sufficient for the action range for solving the pendumlum.
For the other features we enable to have different number of bins per feature and different support possible dense locations of the bins:
* equal: equal sized bins
* center: More bins at the center of the feature space
* edge: More bins at the edges of the feature space
* start: 
* end: Most bins are the 
There's a scaling parameter which defines how much bigger 

For the pendulum more bins at the center leads to better results because this helps balancing.

# Pendulum v-0

The runtime of the full script including training the Neural Networks and executing the value iteration takes 5 minutes and 12 seconds.
 
![pendulum 0](./Plots/Pendulum/pendulum-v0.gif)


## Value Iteration
* Result for 200 bins which are more dense in the center for both features: 

### Iteration 0
![dynamics state 0](./Plots/Pendulum/ValueIteration/ValueIteration_iter_0.png)
### Iteration 15
![dynamics state 15](./Plots/Pendulum/ValueIteration/ValueIteration_iter_15.png)
### Final Iteration
![dynamics state last](./Plots/Pendulum/ValueIteration/ValueItertation_iter_last.png)

### Resulting Policy
![dynamics state last](./Plots/Pendulum/ValueIteration/ValueIteration_policy.png)

Result for Pendulum-v0
`average reward over 100 episodes:: -144.677 +- 78.414 min: -368.291 max: -1.618`

## Policy Iteration
![dynamics state last](./Plots/Pendulum/PolicyIteration/PolicyIteration_value_function.png)
![dynamics state last](./Plots/Pendulum/PolicyIteration/PolicyIteration_policy.png)

[Sourcecode](./main.py#L10)

`average reward over 100 episodes: -149.038 +- 79.486 min: -356.445 max: -2.943`

### Distribution over states using Monte Carlo Sampling

We tried both normal and uniform distributions over monte carlo samples

 
 ### 
 # Qube-v0
 
 For changing to Qube-v0 to you must set `env_name = 'Qube-v0'` in [challenge1.py](./challenge1.py#L18).
  
 ![pendulum 0](./Plots/Qube/qube-v0.gif)

 Taking large action ranges leads to invalid states because Dynamic Programming can't take this into account by default.
 Interpolating
 
 A stochastic version using a multinomal distribution should lead to better results, however the 
 
 `average reward over 100 episodes: -36.464 +- 3.562 min: -46.296 max: -26.041`
 
 ## Filter
 ### Median Filter & Gaussian Smoothing
 
 If we a have an ok policy it's beneficial to apply a median filter to reduce noise.
 For a very good policy applying a median filter on the policy matrix doesn't give any benefits.
 