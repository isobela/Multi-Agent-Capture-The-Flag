from make_env import make_env
import numpy as np

# Create and reset environment
env = make_env("capture_the_flag")
env.reset()

world = env.world

# Get blue seeker 0 and red flag
seeker = world.agents[0]
red_flag = world.landmarks[1]

print("Starting test...")
print("Seeker:", seeker.name)

# STEP 1: Move seeker to red flag
seeker.state.p_pos = red_flag.state.p_pos.copy()

# Run game logic
world.game_update_callback(world)

# Check if seeker picked up flag
assert seeker.hasFlag == True
assert red_flag.carrier is seeker

print("PASS: Blue seeker picked up red flag!")

# STEP 2: Move seeker back to blue spawn
seeker.state.p_pos = world.spawn_zones["blue"].copy()

# Run game logic again
world.game_update_callback(world)

# Check whether blue won
assert world.winner == "blue"
assert world.game_over == True

print("PASS: Blue team captured the flag!")

# STEP 3: Check rewards
rewards = [env._get_reward(agent) for agent in world.agents]

assert rewards[:4] == [1.0] * 4
assert rewards[4:] == [0.0] * 4

print("PASS: Shared team reward works!")

# STEP 4: Check episode termination
dones = [env._get_done(agent) for agent in world.agents]

assert all(dones)

print("PASS: Episode terminates correctly!")

env.close()

print("\nAll tests passed!")