
import numpy as np
import sys
import copy
import time
import random
import argparse
import math

class UCBAgent:
    def __init__(self):
        self.name = "Uma the UCB Agent"
        self.arms = None
        self.total_pulls = 0
        self.c = 2  # exploration parameter, can be adjusted

    def recommendArm(self, bandit, history):
        num_arms = bandit.getNumArms()

        # Initialize arms if this is the first call
        if self.arms is None:
            self.arms = [{"mean": 0, "N": 0} for _ in range(num_arms)]

        # Update arms based on history
        for arm, reward in history:
            self.update(arm, reward)

        self.total_pulls += 1

        # If there are any arms that haven't been pulled, pull them first
        for i, arm in enumerate(self.arms):
            if arm["N"] == 0:
                return i

        # Calculate UCB for each arm and select the best
        ucb_values = []
        for arm in self.arms:
            ucb = arm["mean"] + self.c * math.sqrt(math.log(self.total_pulls) / arm["N"])
            ucb_values.append(ucb)

        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.arms[arm]["N"] += 1
        n = self.arms[arm]["N"]
        current_mean = self.arms[arm]["mean"]
        self.arms[arm]["mean"] = ((n - 1) / n) * current_mean + (1 / n) * reward
