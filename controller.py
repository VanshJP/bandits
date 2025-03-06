import numpy as np
import sys
import copy
import time
import os
import random
from randAgent import randomAgent
from epsGreedyAgent import epsGreedyAgent
from UCBAgent import UCBAgent
from thompsonAgent import thompsonAgent
import argparse
import matplotlib.pyplot as plt


######################################################
AGENTS_MAP = {'randomAgent': randomAgent,
              'epsGreedyAgent': epsGreedyAgent,
              'UCBAgent': UCBAgent,
              'thompsonAgent': thompsonAgent}

class bandit:
    def __init__(self, file):
        f = open(file, "r")
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip("\n")
        self.arms = []
        for i in range(1, len(lines)):
            self.arms.append(float(lines[i]))
        
    def pull_arm(self, arm):
        prob = self.arms[arm]
        randNum = random.random()
        if randNum <= prob:
            return 1
        else:
            return 0

    def getNumArms(self):
        return len(self.arms)

    def getMaxExpectedReward(self):
        return max(self.arms)  # Maximum expected reward of the best arm


def run_experiment(bandit, agent_class, num_plays):
    agent = agent_class()
    history = []
    cumulative_rewards = np.zeros(num_plays)
    regrets = np.zeros(num_plays)
    max_expected_reward = bandit.getMaxExpectedReward()  # Best possible reward
    
    for i in range(num_plays):
        testArm = agent.recommendArm(bandit, history)
        reward = bandit.pull_arm(testArm)
        
        cumulative_rewards[i] = reward if i == 0 else cumulative_rewards[i-1] + reward
        
        # Calculate regret: Max possible expected reward - expected reward of chosen arm
        regret = max_expected_reward - bandit.arms[testArm]
        regrets[i] = regret if i == 0 else regrets[i-1] + regret
        
        history.append((testArm, reward))
    
    return cumulative_rewards, regrets


def main():
    parser = argparse.ArgumentParser(description='Define bandit problem and agents.')
    parser.add_argument('--input', choices=['bandits/input/test0.txt', 'bandits/input/test1.txt'], default='bandits/input/test1.txt', help='The input file, can be input/test0.txt or input/test1.txt')
    parser.add_argument('--num_plays', type=int, default=10000, help='The number of pulls an agent has.')
    args = parser.parse_args()

    testBandit = bandit(args.input)

    # Run experiments for all agents
    cumulative_rewards_results = {}
    regret_results = {}
    
    for agent_name, agent_class in AGENTS_MAP.items():
        cumulative_rewards, regrets = run_experiment(testBandit, agent_class, args.num_plays)
        cumulative_rewards_results[agent_name] = cumulative_rewards
        regret_results[agent_name] = regrets

    # Plotting cumulative rewards
    plt.figure(figsize=(12, 8))
    for agent_name, rewards in cumulative_rewards_results.items():
        plt.plot(range(0, args.num_plays), rewards, label=f"{agent_name} - Rewards")

    plt.xlabel("Number of Pulls")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Comparison of Cumulative Rewards for Bandit Algorithms ({args.input})")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, args.num_plays)  # Set x-axis from 0 to num_plays
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.savefig(f"bandit_rewards_comparison_{os.path.basename(args.input)}.png")
    plt.show()

    # Plotting regrets
    plt.figure(figsize=(12, 8))
    for agent_name, regrets in regret_results.items():
        plt.plot(range(0, args.num_plays), regrets, label=f"{agent_name} - Regret")

    plt.xlabel("Number of Pulls")
    plt.ylabel("Total Regret")
    plt.title(f"Comparison of Regret for Bandit Algorithms ({args.input})")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, args.num_plays)  # Set x-axis from 0 to num_plays
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    plt.savefig(f"bandit_regret_comparison_{os.path.basename(args.input)}.png")
    plt.show()

    # Print final cumulative rewards and regrets
    for agent_name, rewards in cumulative_rewards_results.items():
        print(f"{agent_name} - Final Cumulative Reward: {rewards[-1]}")
    
    for agent_name, regrets in regret_results.items():
        print(f"{agent_name} - Final Total Regret: {regrets[-1]}")


if __name__ == "__main__":
    main()
