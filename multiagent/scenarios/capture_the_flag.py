from multiagent.core import World, Agent, Landmark

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 0 # no communication

        num_blue_agents = 4
        num_red_agents = 4
        num_agents = num_blue_agents + num_red_agents
        num_landmarks = 2 # 2 flags

        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.team = 0 if i < 4 else 1 # team blue (i=0,1,3,4), team red (i=4,5,6,7)
            agent.role = "seeker" if i % 4 < 2 else "guard" # i = 0,1,4,5 are seekers & i = 2,3,6,7 are guards
            




