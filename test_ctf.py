from make_env import make_env
import numpy as np
import time

env = make_env("capture_the_flag")
observations = env.reset()

print("Number of agents:", env.n)
print("Starting Capture the Flag simulation!")

for step in range(200):

    actions = []

    for i in range(env.n):

        # Random movement: stay, right, left, up, down
        action_index = np.random.randint(0, 5)

        # Convert discrete action to one-hot vector
        action = np.zeros(5)
        action[action_index] = 1

        actions.append(action)

    observations, rewards, dones, infos = env.step(actions)

    env.render(mode="human")
    time.sleep(0.05)

    if all(dones):
        print("Winning team:", env.world.winner)
        print("Final rewards:", rewards)
        break

env.close()
