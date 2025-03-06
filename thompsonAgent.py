import numpy as np
import sys
import copy
import time
import random
import argparse

class thompsonAgent: 
    def __init__(self):
        self.name = "Terry the Thompson Sampling Agent"
        self.arms = None

    def recommendArm(self, bandit, history):
        if self.arms is None:
            num_arms = bandit.getNumArms()
            self.arms = [{"alpha": 1, "beta": 1} for _ in range(num_arms)]
        
        if history:
            last_arm, last_reward = history[-1]
            self.update(last_arm, last_reward)

        samples = [np.random.beta(arm["alpha"], arm["beta"]) for arm in self.arms]

        max_sample = max(samples)
        best_arms = [i for i, sample in enumerate(samples) if sample == max_sample]
        chosen_arm = random.choice(best_arms)

        return chosen_arm

    def update(self, arm, reward):
        if reward == 1:
            self.arms[arm]["alpha"] += 1
        else:
            self.arms[arm]["beta"] += 1
