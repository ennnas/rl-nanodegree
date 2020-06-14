[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This repository includes the code needed to train the agent to solve Udacity navigation project based 
on Unity's Banana Collector environment. 

### Environment 

The agent operates in a large, square world. 
The agent in this project interacts with an observation space of size 37 through the selection of one 
of 4 actions.
The state space contains information like the agent's velocity, along with ray-based perception of objects around agent's forward direction.  
Given this information, the agent has to learn how to best select actions.
The allowed actions are:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and the environment is considered solved if the agent gets an average score of +13 over 100 consecutive episodes.


![Trained Agent][image1]

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the current folder and unzip (or decompress) the file. 

## Development setup

To run all the code available in this repo you need to install the following software:

* Python = 3.6 (other versions lead to compatibility issue)
* [poetry](https://python-poetry.org/) ≥ 1.0

### Poetry installation
To install and configure `poetry` run 
```shell script
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
poetry config virtualenvs.in-project true
```

## Installation

First, to install all the required dependencies run
```shell script
poetry install
```

### Instructions

To train the agent open the [notebook](Navigation.ipynb) and execute all cells.
The Q network of the agent is dumped into the `banana_collector.pth` checkpoint file. 

### Results

A trained model with an average score over 100 episodes of 17.00 is included in this repository.
For a more complete description of the results, refer to the report page.
To visualize the agent, use this [notebook](Visualize.ipynb).