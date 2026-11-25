
import time
import numpy as np

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from models.scripted_policy import scripted_action

# Load Capture the Flag scenario
scenario = scenarios.load("capture_the_flag.py").Scenario()
world = scenario.make_world()

# Create environment
env = MultiAgentEnv(
    world,
    reset_callback=scenario.reset_world,
    reward_callback=scenario.reward,
    observation_callback=scenario.observation,
    done_callback=scenario.done,
    shared_viewer=True,
)

# Disable global reward sharing
env.shared_reward = False

# Reset
obs_n = env.reset()

# Run scripted vs scripted
for step in range(300):

    action_n = []

    for i, agent in enumerate(env.agents):
        action = scripted_action(agent, env.world)

        # Convert discrete action to one-hot if required
        if env.discrete_action_input:
            action_n.append(action)
        else:
            one_hot = np.zeros(env.action_space[i].n)
            one_hot[action] = 1.0
            action_n.append(one_hot)
    
    if step < 5:
        print(f"\nSTEP {step}")

        for i, agent in enumerate(env.agents):
            print(
                agent.name,
                "action:", action_n[i],
                "position:", agent.state.p_pos.copy()
            )

    obs_n, reward_n, done_n, info_n = env.step(action_n)

    if step < 5:
        for agent in env.agents:
            print(
                agent.name,
                "NEW position:", agent.state.p_pos,
                "velocity:", agent.state.p_vel,
                "force:", agent.action.u
            )

    env.render()
    time.sleep(0.03)

    if step % 25 == 0:
        print(f"Step {step}")
        for flag in env.world.landmarks:
            carrier = flag.carrier.name if flag.carrier else "None"
            print(f"  {flag.name}: carrier={carrier}")

    if all(done_n):
        print(f"Game finished at step {step}")
        print(f"Winner: {env.world.winner}")
        print(f"Reward: {reward_n}")
        break

else:
    print("Maximum steps reached - draw")

env.close()
