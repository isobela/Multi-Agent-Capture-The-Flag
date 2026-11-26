import os
import csv
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios.capture_the_flag as capture_the_flag

from models.actor import Actor
from models.critic import Two_Head_Critic

from models.scripted_policy import scripted_action

def make_train_env(reward_mode, train_just_seekers=False):
    scenario = capture_the_flag.Scenario()
    world = scenario.make_world()
    world.reward_mode = reward_mode
    if train_just_seekers == True:
        world.agents = [agent for agent in world.agents if agent.role == "seeker"]
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.done)
    env.discrete_action_input = True
    return env, world

def returns_and_advantages(rewards, values, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.FloatTensor(returns)
    values = torch.cat(values).reshape(-1)

    if len(returns)>1:
        advantages = returns - values.detach()
    else:
        advantages = torch.zeros_like(returns)

    return returns, advantages

def stage_two_seeker_observation(env, observations, agent_index):
    observation = observations[agent_index]
    agent = env.agents[agent_index]
    other_agents = [other for other in env.world.agents if other is not agent]

    seeker_positions = []
    guard_positions = []
    for i, other in enumerate(other_agents):
        relative_position = observation[4 + 2 * i:6 + 2 * i]
        if other.role == "seeker":
            seeker_positions.append(relative_position)
        else:
            guard_positions.append(relative_position)

    landmarks_start = 4 + 2 * len(other_agents)
    return np.concatenate(
        [observation[:4], *seeker_positions, *guard_positions, observation[landmarks_start:]]
    )

def expand_actor_input(actor, input_dim, prefix_dim, suffix_dim):
    old_input_dim = actor.fc1.in_features
    if old_input_dim == input_dim:
        return actor

    old_suffix_start = old_input_dim - suffix_dim
    new_suffix_start = input_dim - suffix_dim
    if old_suffix_start != prefix_dim or new_suffix_start < prefix_dim:
        raise ValueError(
            f"Cannot map actor inputs: expected prefix/suffix split at {prefix_dim}, "
            f"got actor input size {old_input_dim} and new input size {input_dim}."
        )

    expanded_fc1 = nn.Linear(input_dim, actor.fc1.out_features).to(
        device=actor.fc1.weight.device, dtype=actor.fc1.weight.dtype
    )
    with torch.no_grad():
        expanded_fc1.weight.zero_()
        expanded_fc1.weight[:, :prefix_dim].copy_(actor.fc1.weight[:, :prefix_dim])
        expanded_fc1.weight[:, new_suffix_start:].copy_(actor.fc1.weight[:, old_suffix_start:])
        expanded_fc1.bias.copy_(actor.fc1.bias)

    actor.fc1 = expanded_fc1
    return actor

def map_action_for_env(agent, action):
    if agent.team == "red" and action in (1,2):
        return 2 if action == 1 else 1
    return action

#---train seeker without opponents/guards
def train_seekers(reward_mode):
    env, world = make_train_env(reward_mode, train_just_seekers=True)
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    num_agents = len(env.agents)
    global_state_dim = obs_dim*num_agents

    seeker_actor = Actor(obs_dim, action_dim)
    critic = Two_Head_Critic(global_state_dim)

    seeker_actor_optimizer = optim.Adam(seeker_actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99

    blue_win_history = deque(maxlen=50)
    red_win_history = deque(maxlen=50)
    draw_history = deque(maxlen=50)
    blue_pickup_history = deque(maxlen=50)
    red_pickup_history = deque(maxlen=50)
    steps_to_capture_history = deque(maxlen=50)

    log_file = open("Multi-Agent-Capture-The-Flag/metrics_stage1.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        ["episode", 
        "total_capture_rate", 
        "blue_win_rate", 
        "red_win_rate",
        "draw_rate",
        "blue_pickups",
        "red_pickups",
        "avg_blue_pickups",
        "avg_red_pickups",
        "avg_steps_to_capture"])

    for episode in range(1000):
        obs = env.reset()   

        episode_rewards = [[] for _ in range(num_agents)]
        episode_log_probs = [[] for _ in range(num_agents)]
        episode_entropies = [[] for _ in range(num_agents)]
        episode_values = [[] for _ in range(num_agents)]
        
        steps_taken = 0

        for step in range(200):
            steps_taken = step + 1

            global_state = torch.FloatTensor(np.concatenate(obs)).unsqueeze(0)
            val_seeker, _ = critic(global_state)

            actions = []
            for i, agent in enumerate(env.agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                dist = seeker_actor(obs_tensor)
                action = dist.sample().item()
                actions.append(map_action_for_env(agent, action))

                episode_log_probs[i].append(dist.log_prob(torch.tensor([action], dtype=torch.long)).unsqueeze(0))
                episode_entropies[i].append(dist.entropy().unsqueeze(0))
                episode_values[i].append(val_seeker)

            next_obs, rewards, dones, info = env.step(actions)

            for i in range(num_agents):
                episode_rewards[i].append(rewards[i])

            obs = next_obs

            if all(dones) or getattr(world, 'game_over', False):
                break

        # Update the policy and critic once, after all episode graphs are collected.
        seeker_log_probs = []
        for i in range(num_agents):
            if len(episode_rewards[i]) == 0:
                continue

            returns, advantages = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)

            for log_p, entropy, adv in zip(episode_log_probs[i], episode_entropies[i], advantages):
                policy_loss = -log_p * adv.item() - 0.01*entropy
                seeker_log_probs.append(policy_loss)

        if len(seeker_log_probs) > 0:
            seeker_actor_optimizer.zero_grad()
            seeker_policy_loss = torch.stack(seeker_log_probs).sum()
            seeker_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(seeker_actor.parameters(), max_norm=0.5)
            seeker_actor_optimizer.step()

        values, returns = [], []
        for i in range(num_agents):
            if len(episode_rewards[i]) == 0:
                continue

            ret, _ = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            returns.append(ret)
            values.append(torch.cat(episode_values[i]).reshape(-1))

        if len(values) > 0:
            val_tensor = torch.cat(values)
            ret_tensor = torch.cat(returns)
            critic_loss = nn.MSELoss()(val_tensor, ret_tensor)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            critic_optimizer.step()

        #record episode stats
        winner = getattr(world, "winner", None)
        
        blue_win_history.append(1.0 if winner == "blue" else 0.0)
        blue_pickups = world.pickup_count["blue"]
        blue_pickup_history.append(blue_pickups)
        
        red_win_history.append(1.0 if winner == "red" else 0.0)
        red_pickups = world.pickup_count["red"]
        red_pickup_history.append(red_pickups)

        draw_history.append(1.0 if winner is None else 0.0)

        if winner is not None:
            steps_to_capture_history.append(steps_taken)

        blue_wr = np.mean(blue_win_history)*100
        red_wr = np.mean(red_win_history)*100
        draw_rate = np.mean(draw_history)*100
        total_cr = blue_wr + red_wr #total % of episodes ending in a capture    
        
        avg_steps_to_capture = np.mean(steps_to_capture_history) if steps_to_capture_history else 200.0
        
        avg_blue_pickups = np.mean(blue_pickup_history)
        avg_red_pickups = np.mean(red_pickup_history)
        
        writer.writerow([episode + 1, total_cr, blue_wr, red_wr, draw_rate, blue_pickups, red_pickups, avg_blue_pickups, avg_red_pickups, avg_steps_to_capture])    

        if (episode + 1) % 50 == 0:
            total_blue_pickups_50 = int(sum(blue_pickup_history))
            total_red_pickups_50 = int(sum(red_pickup_history))
            print(
                f"Ep {episode+1} | Total Capture Rate: {total_cr:.1f}% | "
                f"Pickups In the Last 50 Eps (B/R): {total_blue_pickups_50}/{total_red_pickups_50} |"
                f"Avg Blue Pickups: {avg_blue_pickups:.2f} | Avg Red Pickups: {avg_red_pickups:.2} |"
                f"Blue Win Rate: {blue_wr:.1f}% | Red Win Rate: {red_wr:.1f}% | Avg Steps To Capture: {avg_steps_to_capture:.1f}"
            )

    log_file.close()        

    return seeker_actor

def train_with_static_guards(reward_mode, trained_seeker_actor=None):
    env, world = make_train_env(reward_mode)
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    num_agents = len(env.agents)
    global_state_dim = obs_dim*num_agents

    if trained_seeker_actor is None:
        seeker_actor = Actor(obs_dim, action_dim)
    else:
        prefix_dim = 4 + 2 * (sum(agent.role == "seeker" for agent in env.agents) - 1)
        suffix_dim = 2 * len(world.landmarks) + 4
        seeker_actor = expand_actor_input(
            trained_seeker_actor, obs_dim, prefix_dim, suffix_dim
        )
    critic = Two_Head_Critic(global_state_dim)

    seeker_actor_optimizer = optim.Adam(seeker_actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99

    blue_win_history = deque(maxlen=50)
    red_win_history = deque(maxlen=50)
    draw_history = deque(maxlen=50)
    blue_pickup_history = deque(maxlen=50)
    red_pickup_history = deque(maxlen=50)
    blue_tag_history = deque(maxlen=50)
    red_tag_history = deque(maxlen=50)
    steps_to_capture_history = deque(maxlen=50)
    
    log_file = open("Multi-Agent-Capture-The-Flag/metrics_stage2.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        ["episode", 
        "total_capture_rate", 
        "blue_win_rate", 
        "red_win_rate",
        "draw_rate", 
        "blue_pickups",
        "red_pickups",
        "avg_blue_pickups",
        "avg_red_pickups",
        "avg_blue_tags",
        "avg_red_tags",
        "avg_steps_to_capture"])

    for episode in range(1000):
        obs = env.reset()

        for agent in env.agents:
            if agent.role == "guard":
                x_pos = -0.25 if agent.team == "blue" else 0.25
                x_pos += np.random.uniform(-0.2, 0.2)
                agent.state.p_pos = np.array([x_pos, 0.0])
        obs = [env._get_obs(agent) for agent in env.agents]     

        episode_rewards = [[] for _ in range(num_agents)]
        episode_log_probs = [[] for _ in range(num_agents)]
        episode_entropies = [[] for _ in range(num_agents)]
        episode_values = [[] for _ in range(num_agents)]

        steps_taken = 0

        for step in range(200):

            steps_taken = step + 1

            global_state = torch.FloatTensor(np.concatenate(obs)).unsqueeze(0)
            val_seeker, _ = critic(global_state)

            actions = []
            for i, agent in enumerate(env.agents):
                if agent.role == "guard":
                    actions.append(0)
                    continue  

                actor_obs = stage_two_seeker_observation(env, obs, i)
                obs_tensor = torch.FloatTensor(actor_obs).unsqueeze(0)
                dist = seeker_actor(obs_tensor)
                action = dist.sample().item()
                actions.append(map_action_for_env(agent,action))

                episode_log_probs[i].append(dist.log_prob(torch.tensor([action], dtype=torch.long)).unsqueeze(0))
                episode_entropies[i].append(dist.entropy().unsqueeze(0))
                episode_values[i].append(val_seeker)

            next_obs, rewards, dones, info = env.step(actions)

            for i in range(num_agents):
                episode_rewards[i].append(rewards[i])
                
            obs = next_obs

            if all(dones) or getattr(world, 'game_over', False):
                break

        seeker_log_probs = []
        for i in range(num_agents):
            if env.agents[i].role == "guard" or len(episode_rewards[i]) == 0:
                continue  # Skip guards for policy updates
            _, advantages = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            for log_p, entropy, adv in zip(episode_log_probs[i], episode_entropies[i], advantages):
                policy_loss = -log_p * adv.item() - 0.01*entropy
                seeker_log_probs.append(policy_loss)

        if len(seeker_log_probs) > 0:
            seeker_actor_optimizer.zero_grad()
            seeker_policy_loss = torch.stack(seeker_log_probs).sum()
            seeker_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(seeker_actor.parameters(), max_norm=0.5)
            seeker_actor_optimizer.step()

        values, returns = [], []
        for i in range(num_agents):
            if env.agents[i].role == "guard" or len(episode_rewards[i]) == 0:
                continue
            ret, _ = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            returns.append(ret)
            values.append(torch.cat(episode_values[i]).reshape(-1))

        if(len(values) > 0):
            val_tensor = torch.cat(values)
            ret_tensor = torch.cat(returns)
            critic_loss = nn.MSELoss()(val_tensor, ret_tensor)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm = 0.5)
            critic_optimizer.step()

        winner = getattr(world, "winner", None)
        blue_win_history.append(1.0 if winner == "blue" else 0.0)
        red_win_history.append(1.0 if winner == "red" else 0.0)

        draw_history.append(1.0 if winner == None else 0.0)

        blue_pickups = world.pickup_count["blue"]
        blue_pickup_history.append(blue_pickups)
        
        red_pickups = world.pickup_count["red"]
        red_pickup_history.append(red_pickups)
        
        blue_tag_history.append(world.tag_count["blue"])
        red_tag_history.append(world.tag_count["red"])

        if winner is not None:
            steps_to_capture_history.append(steps_taken)    

        blue_wr = np.mean(blue_win_history)*100
        red_wr = np.mean(red_win_history)*100
        draw_rate = np.mean(draw_history)*100
        total_cr = blue_wr + red_wr #total % of episodes ending in a capture    
        avg_steps_to_capture = np.mean(steps_to_capture_history) if steps_to_capture_history else 200.0
        avg_blue_pickups = np.mean(blue_pickup_history)
        avg_red_pickups = np.mean(red_pickup_history)
        avg_blue_tags = np.mean(blue_tag_history)
        avg_red_tags = np.mean(red_tag_history)
        
        writer.writerow([episode + 1, total_cr, blue_wr, red_wr, draw_rate, blue_pickups, red_pickups, avg_blue_pickups, avg_red_pickups, avg_blue_tags, avg_red_tags, avg_steps_to_capture])    
    
        if (episode + 1) % 50 == 0:
            total_blue_pickups_50 = int(sum(blue_pickup_history))
            total_red_pickups_50 = int(sum(red_pickup_history))
            print(
                f"Ep {episode+1} | Total Capture Rate: {total_cr:.1f}% | "
                f"Pickups In the Last 50 Eps (B/R): {total_blue_pickups_50}/{total_red_pickups_50} |"
                f"Avg Blue Pickups In Past 50 Eps: {avg_blue_pickups:.2f} | Avg Red Pickups In Past 50 Eps: {avg_red_pickups:.2} |"
                f"Avg Blue Tags In Past 50 Eps: {avg_blue_tags:.2f} | Avg Red Tags In Past 50 Eps: {avg_red_tags:.2f} |"
                f"Blue Win Rate: {blue_wr:.1f}% | Red Win Rate: {red_wr:.1f}% | Avg Steps To Capture: {avg_steps_to_capture:.1f}"
            )    
        
    log_file.close()       

    return seeker_actor

#train guards with trained seekers
def train_guards_with_trained_seekers(reward_mode, trained_seeker_actor):
    env, world = make_train_env(reward_mode)
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    num_agents = len(env.agents)
    global_state_dim = obs_dim*num_agents
    
    trained_seeker_actor.eval()
    for param in trained_seeker_actor.parameters():
        param.requires_grad = False

    guard_actor = Actor(obs_dim, action_dim)
    critic = Two_Head_Critic(global_state_dim)

    guard_actor_optimizer = optim.Adam(guard_actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99

    blue_win_history = deque(maxlen=50)
    red_win_history = deque(maxlen=50)
    draw_history = deque(maxlen=50)
    blue_pickup_history = deque(maxlen=50)
    red_pickup_history = deque(maxlen=50)
    blue_tag_history = deque(maxlen=50)
    red_tag_history = deque(maxlen=50)
    steps_to_capture_history = deque(maxlen=50)
        
    log_file = open("Multi-Agent-Capture-The-Flag/metrics_stage3.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        ["episode", 
        "total_capture_rate", 
        "blue_win_rate", 
        "red_win_rate",
        "draw_rate", 
        "blue_pickups",
        "red_pickups",
        "avg_blue_pickups",
        "avg_red_pickups",
        "avg_blue_tags",
        "avg_red_tags",
        "avg_steps_to_capture"])

    for episode in range(1000):
        obs = env.reset()

        episode_rewards = [[] for _ in range(num_agents)]
        episode_log_probs = [[] for _ in range(num_agents)]
        episode_entropies = [[] for _ in range(num_agents)]
        episode_values = [[] for _ in range(num_agents)]

        steps_taken = 0

        for step in range(200):
            steps_taken = step + 1

            global_state = torch.FloatTensor(np.concatenate(obs)).unsqueeze(0)
            _, val_guard = critic(global_state)

            actions = []
            for i, agent in enumerate(env.agents):
                #blue and red guards learn
                if agent.role == "guard":
                    obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                    dist = guard_actor(obs_tensor)
                    action = dist.sample().item()
                    actions.append(map_action_for_env(agent, action))

                    episode_log_probs[i].append(dist.log_prob(torch.tensor([action], dtype=torch.long)).unsqueeze(0))
                    episode_entropies[i].append(dist.entropy().unsqueeze(0))
                    episode_values[i].append(val_guard)
                else:
                    #blue and red seekers run stage 2 policy
                    with torch.no_grad():
                        actor_obs = stage_two_seeker_observation(env, obs, i)
                        obs_tensor = torch.FloatTensor(actor_obs).unsqueeze(0)
                        dist = trained_seeker_actor(obs_tensor)
                        action = dist.sample().item()
                    actions.append(map_action_for_env(agent, action))

            next_obs, rewards, dones, info = env.step(actions)

            for i, agent in enumerate(env.agents):
                if agent.role == "guard":
                    episode_rewards[i].append(rewards[i])

            obs = next_obs

            if all(dones) or getattr(world, "game_over", False):
                break

        guard_log_probs = []
        for i, agent in enumerate(env.agents):
            if agent.role == "seeker" or len(episode_rewards[i]) == 0:
                continue
            _, advantages = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            for log_p, entropy, adv in zip(episode_log_probs[i], episode_entropies[i], advantages):
                policy_loss = -log_p * adv.item() - 0.01*entropy
                guard_log_probs.append(policy_loss)

        if len(guard_log_probs) > 0:
            guard_actor_optimizer.zero_grad()
            guard_policy_loss = torch.stack(guard_log_probs).sum()
            guard_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(guard_actor.parameters(), max_norm=0.5)
            guard_actor_optimizer.step()

        values, returns = [], []
        for i, agent in enumerate(env.agents):
            if agent.role == "seeker" or len(episode_rewards[i]) == 0:
                continue
            ret, _ = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            returns.append(ret)
            values.append(torch.cat(episode_values[i]).reshape(-1))

        if len(values) > 0:
            val_tensor = torch.cat(values)
            ret_tensor = torch.cat(returns)
            critic_loss = nn.MSELoss()(val_tensor, ret_tensor)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm = 0.5)
            critic_optimizer.step()

        winner = getattr(world, "winner", None)
        blue_win_history.append(1.0 if winner == "blue" else 0.0)
        red_win_history.append(1.0 if winner == "red" else 0.0)

        draw_history.append(1.0 if winner == None else 0.0)

        blue_pickups = world.pickup_count["blue"]
        blue_pickup_history.append(blue_pickups)
        
        red_pickups = world.pickup_count["red"]
        red_pickup_history.append(red_pickups)
        
        blue_tag_history.append(world.tag_count["blue"])
        red_tag_history.append(world.tag_count["red"])

        if winner is not None:
            steps_to_capture_history.append(steps_taken)    

        blue_wr = np.mean(blue_win_history)*100
        red_wr = np.mean(red_win_history)*100
        draw_rate = np.mean(draw_history)*100
        total_cr = blue_wr + red_wr #total % of episodes ending in a capture    
        avg_steps_to_capture = np.mean(steps_to_capture_history) if steps_to_capture_history else 200.0
        avg_blue_pickups = np.mean(blue_pickup_history)
        avg_red_pickups = np.mean(red_pickup_history)
        avg_blue_tags = np.mean(blue_tag_history)
        avg_red_tags = np.mean(red_tag_history)
        
        writer.writerow([episode + 1, total_cr, blue_wr, red_wr, draw_rate, blue_pickups, red_pickups, avg_blue_pickups, avg_red_pickups, avg_blue_tags, avg_red_tags, avg_steps_to_capture])    
    
        if (episode + 1) % 50 == 0:
            total_blue_pickups_50 = int(sum(blue_pickup_history))
            total_red_pickups_50 = int(sum(red_pickup_history))
            print(
                f"Ep {episode+1} | Total Capture Rate: {total_cr:.1f}% | "
                f"Pickups In the Last 50 Eps (B/R): {total_blue_pickups_50}/{total_red_pickups_50} |"
                f"Avg Blue Pickups In Past 50 Eps: {avg_blue_pickups:.2f} | Avg Red Pickups In Past 50 Eps: {avg_red_pickups:.2} |"
                f"Avg Blue Tags In Past 50 Eps: {avg_blue_tags:.2f} | Avg Red Tags In Past 50 Eps: {avg_red_tags:.2f} |"
                f"Blue Win Rate: {blue_wr:.1f}% | Red Win Rate: {red_wr:.1f}% | Avg Steps To Capture: {avg_steps_to_capture:.1f}"
            )    
        
    log_file.close()

    for param in trained_seeker_actor.parameters():
        param.requires_grad = True
    trained_seeker_actor.train()

    return guard_actor

        
#train against scripted baseline opposing team
def train_with_scripted_team(reward_mode, seeker_actor, guard_actor):
    
    env, world = make_train_env(reward_mode)
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    num_agents = len(env.agents)
    global_state_dim = obs_dim*num_agents

    seeker_actor.train()
    guard_actor.train()
    critic = Two_Head_Critic(global_state_dim)

    seeker_optimizer = optim.Adam(seeker_actor.parameters(), lr=5e-04)
    guard_optimizer = optim.Adam(guard_actor.parameters(), lr=5e-04)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99

    blue_win_history = deque(maxlen=50)
    red_win_history = deque(maxlen=50)
    draw_history = deque(maxlen=50)
    blue_pickup_history = deque(maxlen=50)
    red_pickup_history = deque(maxlen=50)
    blue_tag_history = deque(maxlen=50)
    red_tag_history = deque(maxlen=50)
    steps_to_capture_history = deque(maxlen=50)
    
    log_file = open("Multi-Agent-Capture-The-Flag/metrics_stage4.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        ["episode", 
        "total_capture_rate", 
        "blue_win_rate", 
        "red_win_rate",
        "draw_rate", 
        "blue_pickups",
        "red_pickups",
        "avg_blue_pickups",
        "avg_red_pickups",
        "avg_blue_tags",
        "avg_red_tags",
        "avg_steps_to_capture"])

    for episode in range(1000):
        obs = env.reset()

        episode_rewards = [[] for _ in range(num_agents)]
        episode_log_probs = [[] for _ in range(num_agents)]
        episode_entropies = [[] for _ in range(num_agents)]
        episode_values = [[] for _ in range(num_agents)]

        steps_taken = 0

        for step in range(200):
            steps_taken = step+1
            global_state = torch.FloatTensor(np.concatenate(obs)).unsqueeze(0)
            val_seeker, val_guard = Two_Head_Critic(global_state)

            actions = []
            for i, agent in enumerate(env.agents):
                if agent.team == "blue":
                    if agent.role == "seeker":
                        actor_obs = stage_two_seeker_observation(env, obs, i)
                        obs_tensor = torch.FloatTensor(actor_obs).unsqueeze(0)
                        dist = seeker_actor(obs_tensor)
                        action = dist.sample().item()
                        actions.append(map_action_for_env(agent, action))

                        episode_log_probs[i].append(dist.log_prob(torch.tensor([action], dtype=torch.long)).unsqueeze(0))
                        episode_entropies[i].append(dist.entropy().unsqueeze(0))
                        episode_values[i].append(val_seeker)
                    else:
                        obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                        dist = guard_actor(obs_tensor)
                        action = dist.sample().item()
                        actions.append(map_action_for_env(agent, action))

                        episode_log_probs[i].append(dist.log_prob(torch.tensor([action], dtype=torch.long)).unsqueeze(0))
                        episode_entropies[i].append(dist.entropy().unsqueeze(0))
                        episode_values[i].append(val_guard)    
                else:
                    actions.append(scripted_action(agent, world))

            next_obs, rewards, dones, info = env.step(actions)

            for i, agent in enumerate(env.agents):
                if agent.team == "blue":
                    episode_rewards[i].append(rewards[i])

            obs = next_obs

            if all(dones) or getattr(world, "game_over", False):
                break

        seeker_log_probs = []
        for i, agent in enumerate(env.agents):
            if agent.team == "red" or agent.role == "guard" or len(episode_rewards[i]) == 0:
               continue
            _, advantages = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            for log_p, entropy, adv in zip(episode_log_probs[i], episode_entropies[i], advantages):
                policy_loss = -log_p * adv.item() - 0.01*entropy
                seeker_log_probs.append(policy_loss)          

        if len(seeker_log_probs) > 0:
            seeker_optimizer.zero_grad()
            seeker_policy_loss = torch.stack(seeker_log_probs).sum()
            seeker_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(seeker_actor.parameters(), max_norm=0.5)
            seeker_optimizer.step()

        guard_log_probs = []
        for i, agent in enumerate(env.agents):
            if agent.team == "red" or agent.role == "seeker" or len(episode_rewards[i]) == 0:
                continue
            _, advantages = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            for log_p, entropy, adv in zip(episode_log_probs[i], episode_entropies[i], advantages):
                policy_loss = -log_p * adv.item() - 0.01*entropy
                guard_log_probs.append(policy_loss)
        
        
        if len(guard_log_probs) > 0:
            guard_optimizer.zero_grad()
            guard_policy_loss = torch.stack(guard_log_probs).sum()
            guard_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(guard_actor.parameters(), max_norm=0.5)
            guard_optimizer.step()

        values, returns = [], []
        for i, agent in enumerate(env.agents):
            if agent.team == "red" or len(episode_rewards[i]) == 0:
                continue
            ret, _ = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            returns.append(ret)
            values.append(torch.cat(episode_values[i]).reshape(-1))
        
        if len(values) > 0:
            val_tensor = torch.cat(values)
            ret_tensor = torch.cat(returns)
            critic_loss = nn.MSELoss()(val_tensor, ret_tensor)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm = 0.5)
            critic_optimizer.step()
        
        winner = getattr(world, "winner", None)
        blue_win_history.append(1.0 if winner == "blue" else 0.0)
        red_win_history.append(1.0 if winner == "red" else 0.0)
        
        draw_history.append(1.0 if winner == None else 0.0)
        
        blue_pickups = world.pickup_count["blue"]
        blue_pickup_history.append(blue_pickups)
        
        red_pickups = world.pickup_count["red"]
        red_pickup_history.append(red_pickups)
        
        blue_tag_history.append(world.tag_count["blue"])
        red_tag_history.append(world.tag_count["red"])
        
        if winner is not None:
            steps_to_capture_history.append(steps_taken)    
        
        blue_wr = np.mean(blue_win_history)*100
        red_wr = np.mean(red_win_history)*100
        draw_rate = np.mean(draw_history)*100
        total_cr = blue_wr + red_wr #total % of episodes ending in a capture    
        avg_steps_to_capture = np.mean(steps_to_capture_history) if steps_to_capture_history else 200.0
        avg_blue_pickups = np.mean(blue_pickup_history)
        avg_red_pickups = np.mean(red_pickup_history)
        avg_blue_tags = np.mean(blue_tag_history)
        avg_red_tags = np.mean(red_tag_history)
        
        writer.writerow([episode + 1, total_cr, blue_wr, red_wr, draw_rate, blue_pickups, red_pickups, avg_blue_pickups, avg_red_pickups, avg_blue_tags, avg_red_tags, avg_steps_to_capture])    
        
        if (episode + 1) % 50 == 0:
            total_blue_pickups_50 = int(sum(blue_pickup_history))
            total_red_pickups_50 = int(sum(red_pickup_history))
            print(
                f"Ep {episode+1} | Total Capture Rate: {total_cr:.1f}% | "
                f"Pickups In the Last 50 Eps (B/R): {total_blue_pickups_50}/{total_red_pickups_50} |"
                f"Avg Blue Pickups In Past 50 Eps: {avg_blue_pickups:.2f} | Avg Red Pickups In Past 50 Eps: {avg_red_pickups:.2} |"
                f"Avg Blue Tags In Past 50 Eps: {avg_blue_tags:.2f} | Avg Red Tags In Past 50 Eps: {avg_red_tags:.2f} |"
                f"Blue Win Rate: {blue_wr:.1f}% | Red Win Rate: {red_wr:.1f}% | Avg Steps To Capture: {avg_steps_to_capture:.1f}"
            )    
        
    log_file.close()

    return seeker_actor, guard_actor
                  
    

#train with self-play (both seekers and guards are learning agents)

def main():
    
    os.makedirs("Multi-Agent-Capture-The-Flag/trained_models", exist_ok=True)
    
    seeker_actor = train_seekers("event")
    torch.save(seeker_actor.state_dict(), "Multi-Agent-Capture-The-Flag/trained_models/cond3/stage1_seeker_actor.pth")

    seeker_actor = train_with_static_guards("event", trained_seeker_actor=seeker_actor)
    torch.save(seeker_actor.state_dict(), "Multi-Agent-Capture-The-Flag/trained_models/cond3/stage2_seeker_actor.pth")

    guard_actor = train_guards_with_trained_seekers("event", trained_seeker_actor=seeker_actor)
    torch.save(guard_actor.state_dict(), "Multi-Agent-Capture-The-Flag/trained_models/cond3/stage3_guard_actor.pth")

    seeker_actor, guard_actor = train_with_scripted_team("event", seeker_actor, guard_actor)
    torch.save(seeker_actor.state_dict(), "Multi-Agent-Capture-The-Flag/trained_models/cond3/stage4_seeker_actor.pth")
    torch.save(guard_actor.state_dict(), "Multi-Agent-Capture-The-Flag/trained_models/cond3/stage4_guard_actor.pth")

if __name__ == "__main__":
    main()
        

