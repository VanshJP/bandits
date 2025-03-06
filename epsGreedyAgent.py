
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
        self.epsilon = 0.1
        self.arm_counts = None
        self.arm_values = None
    
    def recommendArm(self, bandit, history):
        num_arms = bandit.getNumArms()

        if self.arm_counts is None:
            self.arm_counts = np.zeros(num_arms)
            self.arm_values = np.zeros(num_arms)

        for arm, reward in history:
            self.arm_counts[arm] += 1
            n = self.arm_counts[arm]
            value = self.arm_values[arm]
            new_value = ((n - 1) / n) * value + (1 / n) * reward
            self.arm_values[arm] = new_value

        if random.random() < self.epsilon:
            return random.randint(0, num_arms - 1)
        else:
            return np.argmax(self.arm_values)
