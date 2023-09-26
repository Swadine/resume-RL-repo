"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

############################################################################

def KL_DIVG(x, y):
    if x == 0:
        return (1 - x) * math.log((1 - x) / (1 - y)) 
    elif x == 1:
        return float('inf')
    else:
        return x * math.log(x / y) + (1 - x) * math.log((1 - x) / (1 - y)) 

def solve_q(p, c):
    epsilon = 1E-4
    q = p
    b = (1 - p) / 2
    while b > epsilon:
        if KL_DIVG(p, q + b) <= c:
            q += b
        b /= 2
    return q

def KL(values, counts, c, t):
    q = np.zeros(len(values), dtype=np.float32)
    for arm in range(len(values)):
        val = (math.log(t) + c * math.log(math.log(t))) / counts[arm]
        q[arm] = solve_q(values[arm], val)

    return q

############################################################################

class UCB(Algorithm):
    # The algorithm assumes the reward means of the arms of the bandit instance are between 0 and 1
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)   
        self.time_step = 0    
        self.counts = np.zeros(num_arms) 
        self.values = np.ones(num_arms)
    
    def give_pull(self):
        if self.time_step < self.num_arms:
            return self.time_step
        else:
            ucb = self.values + np.sqrt((2 * math.log(self.time_step)) / self.counts)
            return np.argmax(ucb)

    def get_reward(self, arm_index, reward):
        self.time_step += 1
        self.counts[arm_index] += 1
        self.values[arm_index] += ((reward - self.values[arm_index]) / self.counts[arm_index])
        
class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.c = 0 # As mentioned in the papers c increases the upper bound for regret
        self.time_step = 0
        self.counts = np.zeros(num_arms)
        self.values = np.ones(num_arms) # Optmistic Start as I don't know which arm is good and I believe everyone might be good
    
    def give_pull(self):
        if self.time_step < self.num_arms:
            return self.time_step
        else:
            ucb = KL(self.values, self.counts, self.c, self.time_step)
            return np.argmax(ucb)

    def get_reward(self, arm_index, reward):
        self.time_step += 1
        self.counts[arm_index] += 1
        self.values[arm_index] += ((reward - self.values[arm_index]) / self.counts[arm_index])

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.successes = np.zeros(num_arms)
        self.fails = np.zeros(num_arms)
    
    def give_pull(self):
        draws = [np.random.beta(self.successes[i] + 1, self.fails[i] + 1) for i in range(self.num_arms)]
        return np.argmax(draws)
    
    def get_reward(self, arm_index, reward):
        self.successes[arm_index] += reward
        self.fails[arm_index] += (1 - reward)
