import numpy as np
import torch
import torch.optim as optim
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios.capture_the_flag as capture_the_flag

from models.actor import Actor
from models.critic import Critic

def make_train_env():
    scenario = capture_the_flag.Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env, world

