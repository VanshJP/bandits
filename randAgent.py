import numpy as np
import sys
import copy
import time
import random
import argparse

class randomAgent: 
	def __init__(self):
		self.name = "Randy the RandomAgent"
	
	def recommendArm(self, bandit, history):
		numArms = bandit.getNumArms()
		return random.choice(range(numArms))
