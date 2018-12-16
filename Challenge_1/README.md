### Challenge 1

We tested out different model types for learning the dynamics and rewards

## Random Forest

![dynamics state 0](./Plots/Pendulum/Random_Forrest/dynamics_state0.png)
![dynamics state 01](./Plots/Pendulum/Random_Forrest/dynamics_state1.png)
![rewards](./Plots/Pendulum/Random_Forrest/rewards.png)

## Gaussian Process

![dynamics state 0](./Plots/Pendulum/Gaussian_Process/dynamics_state0.png)
![dynamics state 01](./Plots/Pendulum/Gaussian_Process/dynamics_state1.png)
![rewards](./Plots/Pendulum/Gaussian_Process/rewards.png)

## Neural Network

Later on we switched to using a Multi Layer Perceptron with the following architecture:
* 3 hidden layers each having 200 units
* 2 Dropout Layers with 30% rate between last after 2nd and third hidden layer

## Value Iteration
* Result for 200 equal sized bins for both features: 

* X axis: theta dot
* Y axis: theta
### Iteration 0
![dynamics state 0](./Plots/Pendulum/ValueIteration/ValueIteration_iter_0.png)
### Iteration 15
![dynamics state 15](./Plots/Pendulum/ValueIteration/ValueIteration_iter_15.png)
### Final Iteration 15
![dynamics state last](./Plots/Pendulum/ValueIteration/ValueItertation_iter_last.png)

### Resulting Policy
![dynamics state last](./Plots/Pendulum/ValueIteration/ValueIteration_policy.png)

Best results for Pendulum-v2
`100 epochs: Mean reward: -147.71839019523438 -- standard deviation of rewards: 100.89982163365936`

## Policy Iteration
![dynamics state last](./Plots/Pendulum/PolicyIteration/PolicyIteration_value_function.png)
![dynamics state last](./Plots/Pendulum/PolicyIteration/PolicyIteration_policy.png)

[Sourcecode](./main.py#L10)

We found out that if we have a good dynmaic model then it's better to use more bins in the center because
 this helps balaning. 