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
            agent.hasFlag = False

            agent.color = (
                np.array([0.2, 0.4, 0.9]) if agent.team == "blue"
                else np.array([0.9, 0.2, 0.2]))
            
        world.landmarks = [Landmark() for _ in range(2)]

        for i, flag in enumerate(world.landmarks):
            flag.team = "blue" if i == 0 else "red"
            flag.name = f"{flag.team}_flag"

            flag.collide = False
            flag.movable = False # keep false, controlled manually
            flag.size = 0.07
            flag.boundary = False

            flag.carrier = None

        world.spawn_zones = {
            "blue": np.array([-0.7, 0.0]),
            "red": np.array([0.7, 0.0])
        }
        
        world.game_update_callback = self.update_game
        self.reset_world(world)
        return world
    

    def reset_world(self, world):
        world.winner = None
        world.game_over = False
        world.flag_carrier = None
        world.pickup_count = {"blue": 0, "red": 0}
        world.tag_count = {"blue": 0, "red": 0}

        # Initialize flags
        for flag in world.landmarks:
            if flag.team == "blue":
                flag.state.p_pos = np.array([-0.4, 0.0])
                flag.color = np.array([0.2, 0.4, 0.9])
            else:
                flag.state.p_pos = np.array([0.4, 0.0])
                flag.color = np.array([0.9, 0.2, 0.2])

            flag.state.p_vel = np.zeros(world.dim_p)
            flag.carrier = None

        # Spawn agents around their team's spawn zone
        for idx, agent in enumerate(world.agents):

            center = world.spawn_zones[agent.team]

            #give each teammate a distinct vertical offset to avoid spawning on top of each other
            y_offset = -0.15 if (idx % 2 == 0) else 0.15
            # Random offset around spawn center
            offset = np.array([np.random.uniform(-0.08, 0.08), y_offset + np.random.uniform(-0.05, 0.05)])

            agent.state.p_pos = center + offset
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.hasFlag = False

            # Set colors
            if agent.team == "blue":
                agent.color = np.array([0.2, 0.4, 0.9])
            else:
                agent.color = np.array([0.9, 0.2, 0.2])

    def reward(self, agent, world):
        # shared team reward for baseline
        if world.winner is None:
            return 0.0
        return 1.0 if agent.team == world.winner else -1.0

    def observation(self, agent, world):

        #mirror x-axis for red team so both teams see own base on the left (-x)
        sign = -1.0 if agent.team == "red" else 1.0

        def rel_vec(target_pos):
            vec = target_pos - agent.state.p_pos
            return np.array([sign*vec[0], vec[1]])

        own_pos = np.array([sign*agent.state.p_pos[0], agent.state.p_pos[1]])
        own_vel = np.array([sign*agent.state.p_vel[0], agent.state.p_vel[1]])

        teammates = [rel_vec(o.state.p_pos) for o in world.agents if o is not agent and o.team == agent.team]
        opponents = [rel_vec(o.state.p_pos) for o in world.agents if o.team != agent.team]

        own_flag = [rel_vec(f.state.p_pos) for f in world.landmarks if f.team == agent.team]
        enemy_flag = [rel_vec(f.state.p_pos) for f in world.landmarks if f.team != agent.team]
        
        own_home = [rel_vec(world.spawn_zones[agent.team])]

        # Whether this agent is carrying a flag
        has_flag = np.array([float(agent.hasFlag)])

            # Role: seeker = 1, guard = 0
        role = np.array([float(agent.role == "seeker")])

        return np.concatenate(
            [own_pos, own_vel]
            + teammates
            + opponents
            + own_flag
            + enemy_flag
            + own_home
            + [has_flag, role]
        )
    
    def distance(self, entity1, entity2):
        return np.linalg.norm(
            entity1.state.p_pos - entity2.state.p_pos
        )

    def on_territory(self, agent, team):
        x = agent.state.p_pos[0]

        if team == "blue":
            return x < 0
        else:
            return x >= 0

    def respawn(self, agent, world):
        center = world.spawn_zones[agent.team]

        agent.state.p_pos = center + np.random.uniform(
            -0.15, 0.15, size=2
        )
        agent.state.p_vel = np.zeros(world.dim_p)

    def update_game(self, world):
        if world.game_over:
            return
        tagged_seekers = set()

        # Keep agents within the playable area
        for agent in world.agents:
            agent.state.p_pos = np.clip(agent.state.p_pos,-1.0,1.0)

        # 1. Guards tag opposing seekers
        for guard in world.agents:
            if guard.role != "guard":
                continue
            for seeker in world.agents:
                if seeker.role != "seeker":
                    continue
                if seeker.team == guard.team:
                    continue
                # Both must be on the guard's territory
                if not self.on_territory(guard, guard.team):
                    continue
                if not self.on_territory(seeker, guard.team):
                    continue
                if self.distance(guard, seeker) < 0.1:
                    # If seeker carries a flag, return it
                    if seeker.hasFlag:
                        self.drop_flag(world, seeker)

                    world.tag_count[guard.team] +=1    
                    
                    print(f"Tag!!! {guard.name} tagged {seeker.name}")
                    self.respawn(seeker, world)
                    tagged_seekers.add(seeker)

        # 2. Seekers pick up opposing flags
        for seeker in world.agents:
            if seeker.role != "seeker":
                continue
            if seeker in tagged_seekers:
                continue
            if seeker.hasFlag:
                continue
            for flag in world.landmarks:
                if flag.team == seeker.team:
                    continue
                if getattr(flag, "carrier", None) is not None:
                    continue
                if self.distance(seeker, flag) < 0.1:
                    seeker.hasFlag = True
                    flag.carrier = seeker
                    world.pickup_count[seeker.team] += 1

                    print(f"pickup!!! {seeker.name} picked up {flag.name}")

        # 3. Move carried flags and check for capture
        for flag in world.landmarks:
            if getattr(flag, "carrier", None) is None:
                continue
            seeker = flag.carrier
            flag.state.p_pos = seeker.state.p_pos.copy()
            home = world.spawn_zones[seeker.team]
            if np.linalg.norm(seeker.state.p_pos - home) < 0.15:
                world.winner = seeker.team
                world.game_over = True
                print(f"game over {seeker.team.upper()} team wins")
                break
        # update flag position when picked up
        for flag in world.landmarks:
            if flag.carrier is not None:
                flag.state.p_pos = flag.carrier.state.p_pos.copy()

    def drop_flag(self, world, seeker):
        for flag in world.landmarks:
            if flag.carrier is seeker:
                flag.state.p_pos = seeker.state.p_pos.copy()
                flag.state.p_vel = np.zeros(world.dim_p)
                flag.carrier = None

        seeker.hasFlag = False

    def done(self, agent, world):
        return world.game_over

