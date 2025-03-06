
import numpy as np
import sys
import copy
import time
import random
import argparse
######################################################

class epsGreedyAgent: 
    def __init__(self):
        self.name = "Eric the Epsilon Greedy Agent"
        self.epsilon = 0.1  # You can adjust this value
        self.arm_counts = None
        self.arm_values = None
    
    def recommendArm(self, bandit, history):
        #Hey, your code goes here!
        num_arms = bandit.getNumArms() # Give agent info on how many arms there are 

        # Initialize arm counts and values if this is the first call
        if self.arm_counts is None:
            self.arm_counts = np.zeros(num_arms)
            self.arm_values = np.zeros(num_arms)

        # Update arm counts and values based on history
        for arm, reward in history:
            self.arm_counts[arm] += 1
            n = self.arm_counts[arm]
            value = self.arm_values[arm]
            new_value = ((n - 1) / n) * value + (1 / n) * reward
            self.arm_values[arm] = new_value

        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            # Explore: choose a random arm
            return random.randint(0, num_arms - 1)
        else:
            # Exploit: choose the arm with the highest estimated value
            return np.argmax(self.arm_values)
