
from make_env import make_env
import numpy as np
import time

env = make_env("capture_the_flag")
observations = env.reset()

print("Number of agents:", env.n)

for step in range(200):
    actions = []

    for i in range(env.n):
        # Choose one of 5 movement actions
        action_index = np.random.randint(0, 5)

        # Convert to one-hot vector
        action = np.zeros(5)
        action[action_index] = 1

        actions.append(action)

    observations, rewards, dones, infos = env.step(actions)

    env.render()
    time.sleep(0.05)

env.close()

