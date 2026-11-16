import numpy as np

def scripted_action(agent, world):
    if agent.role == "seeker":
        return seeker_policy(agent, world)
    else:
        return guard_policy(agent, world)



def move_toward(agent, target):
    """Return a discrete movement action."""
    delta = target - agent.state.p_pos

    if np.linalg.norm(delta) < 0.05:
        return 0  # Stay

    if abs(delta[0]) > abs(delta[1]):
        return 1 if delta[0] > 0 else 2
    else:
        return 3 if delta[1] > 0 else 4



def seeker_policy(agent, world):
    if agent.hasFlag:
        # Return to own spawn zone
        target = world.spawn_zones[agent.team]
    else:
        # Find enemy flag
        enemy_flag = next(
            flag for flag in world.landmarks
            if flag.team != agent.team
        )
        target = enemy_flag.state.p_pos

    return move_toward(agent, target)

def guard_policy(agent, world):
    enemies = [
        other for other in world.agents
        if other.team != agent.team
        and other.role == "seeker"
    ]

    # Only chase enemies on our territory
    intruders = [
        enemy for enemy in enemies
        if (enemy.state.p_pos[0] < 0 if agent.team == "blue"
            else enemy.state.p_pos[0] >= 0)
    ]

    if intruders:
        closest = min(
            intruders,
            key=lambda enemy: np.linalg.norm(
                enemy.state.p_pos - agent.state.p_pos
            )
        )
        return move_toward(agent, closest.state.p_pos)

    # Otherwise defend own flag
    own_flag = next(
        flag for flag in world.landmarks
        if flag.team == agent.team
    )

    return move_toward(agent, own_flag.state.p_pos)