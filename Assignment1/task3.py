"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np
import math 

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault # probability that the bandit returns a faulty pull

        self.alphas = np.ones((num_arms, 2))
        self.betas = np.ones((num_arms, 2))
        self.context = np.array([1, 0]) # Contextual bandits
    
    def give_pull(self):
        draws = np.dot(np.random.beta(self.alphas, self.betas), self.context)
        return np.argmax(draws)

    def get_reward(self, arm_index, reward):
        self.alphas[arm_index] += reward * self.context
        self.betas[arm_index] += (1 - reward) * self.context
