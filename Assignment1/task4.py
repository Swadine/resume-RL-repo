"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np

class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
        self.alphas = np.ones((num_arms, 2)) # Essentially two features - contextual bandits
        self.betas = np.ones((num_arms, 2))
        # self.context = np.array([1 , 1])
    
    def give_pull(self):
        draws = np.random.beta(self.alphas[:, 0], self.betas[:, 0]) + np.random.beta(self.alphas[:, 1], self.betas[:, 1])
        return np.argmax(draws)
    
    def get_reward(self, arm_index, set_pulled, reward):
        # self.alphas[arm_index] += reward * self.context
        # self.betas[arm_inde] += (1 - reward) * self.context

        self.alphas[arm_index, set_pulled] += reward
        self.betas[arm_index, set_pulled] += (1 - reward)

