from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import numpy as np

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2 # no communication

        num_blue_agents = 4
        num_red_agents = 4
        num_agents = num_blue_agents + num_red_agents
        num_landmarks = 2 # 2 flags

        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.team = "blue" if i < 4 else "red" # team blue (i=0,1,3,4), team red (i=4,5,6,7)
            agent.role = "seeker" if i % 4 < 2 else "guard" # i = 0,1,4,5 are seekers & i = 2,3,6,7 are guards
            agent.name = f"{agent.team} {agent.role} {i}"

            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.0

            agent.color = (
                np.array([0.2, 0.4, 0.9]) if agent.team == "blue"
                else np.array([0.9, 0.2, 0.2]))
            
        world.landmarks = [Landmark() for _ in range(2)]

        for i, flag in enumerate(world.landmarks):
            flag.team = "blue" if i == 0 else "red"
            flag.name = f"{flag.team}_flag"

            flag.collide = False
            flag.movable = False
            flag.size = 0.07
            flag.boundary = False

        world.spawn_zones = {
            "blue": np.array([-0.7, 0.0]),
            "red": np.array([0.7, 0.0])
        }

        self.reset_world(world)
        return world
    

    def reset_world(self, world):

        # Initialize flags
        for flag in world.landmarks:
            if flag.team == "blue":
                flag.state.p_pos = np.array([-0.4, 0.0])
                flag.color = np.array([0.2, 0.4, 0.9])
            else:
                flag.state.p_pos = np.array([0.4, 0.0])
                flag.color = np.array([0.9, 0.2, 0.2])

            flag.state.p_vel = np.zeros(world.dim_p)

        # Spawn agents around their team's spawn zone
        for agent in world.agents:

            center = world.spawn_zones[agent.team]

            # Random offset around spawn center
            offset = np.random.uniform(-0.15, 0.15, size=2)

            agent.state.p_pos = center + offset
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            # Set colors
            if agent.team == "blue":
                agent.color = np.array([0.2, 0.4, 0.9])
            else:
                agent.color = np.array([0.9, 0.2, 0.2])

    def reward(self, agent, world):
        # No rewards yet - just testing the environment
        return 0

    def observation(self, agent, world):
        # Agent observes its own position and velocity
        # plus the positions of all other agents and flags

        other_positions = [
            other.state.p_pos - agent.state.p_pos
            for other in world.agents if other is not agent
        ]

        flag_positions = [
            flag.state.p_pos - agent.state.p_pos
            for flag in world.landmarks
        ]

        return np.concatenate(
            [agent.state.p_pos, agent.state.p_vel]
            + other_positions
            + flag_positions
        )



