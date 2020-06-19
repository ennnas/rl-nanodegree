[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135605-ba0e5f2c-7d12-11e8-9578-86d74e0976f8.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135683-dde5c6f0-7d13-11e8-90b1-8770df3e40cf.gif "Trained Agent"
[cross_entropy]: cross_entropy.png
[hill_climbing]: hill_climbing.png

# Stochastic Policy Search methods

Policy-based methods are a family of methods that learn to estimate the optimal policy
&pi;<top>* without the need of going through the estimation of the optimal action value function Q<top>*(s, a)

Policy-based methods can learn either stochastic or deterministic policies, and they can be used to 
solve environments with either finite or continuous action spaces.This family of methods is particularly suitable when dealing with a continuous action space, since the 
output is already the value or probability of an action. 

Stochastic policy based methods learn the optimal policy &pi;<top>* via current policy perturbation.
The idea is to evaluate slightly different policy and compare the new expected return *J<top>'* with the
one of the current policy and iterate until a certain criteria is met. 

## Cross-Entropy Method
From a biological viewpoint, it is an Evolutionary Algorithm. Some individuals (&pi;<top>') are 
sampled from a population and only the best ones influence the characteristics of future generations.

In the image below individuals are stars and the one chosen for the future are the bold ones.
![cross_entropy]

## Hill Climbing
Hill-Climbing is a simple gradient-free algorithm (i.e., without using the gradient ascent or gradient 
descent method. We try to climb to the top of the curve by only perturbing the current policy.
This method differs from cross-entropy because it evaluates only a single candidate at the time. 
![hill_climbing]

### Instructions

Open `CEM.ipynb` to see an implementation of the cross-entropy method with OpenAI Gym's MountainCarContinuous environment.

Open `Hill_Climbing.ipynb` to see an implementation of hill climbing with adaptive noise scaling with OpenAI Gym's Cartpole environment.


### Results

![Trained Agent][image1]

![Trained Agent][image2]