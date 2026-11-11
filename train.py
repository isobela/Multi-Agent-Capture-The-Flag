import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios.capture_the_flag as capture_the_flag

from models.actor import Actor
from models.critic import Critic

def make_train_env(stage):
    scenario = capture_the_flag.Scenario()
    world = scenario.make_world()
    if stage == 1:
        world.agents = [agent for agent in world.agents if agent.role == "seeker"]
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.done)
    return env, world

def returns_and_advantages(rewards, values, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.FloatTensor(returns)
    values = torch.cat(values).squeeze()

    if len(returns)>1:
        advantages = returns - values.detach()
    else:
        advantages = torch.zeros_like(returns)

    return returns, advantages

#---train seeker without opponents/guards
def train_seekers():
    env, world = make_train_env(stage=1)
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    num_agents = len(env.agents)
    global_state_dim = obs_dim*num_agents

    seeker_actor = Actor(obs_dim, action_dim)
    critic = Critic(global_state_dim)

    seeker_actor_optimizer = optim.Adam(seeker_actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99

    for episode in range(1000):
        obs = env.reset()
        episode_rewards = [[] for _ in range(num_agents)]
        episode_log_probs = [[] for _ in range(num_agents)]
        episode_values = [[] for _ in range(num_agents)]
        total_episode_reward = 0

        for _ in range(100):
            global_state = torch.FloatTensor(np.concatenate(obs)).unsqueeze(0)
            val_seeker, _ = critic(global_state)

            actions = []
            for i in enumerate(env.agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                dist = seeker_actor(obs_tensor)
                action = dist.sample().item()
                actions.append(action)

                episode_log_probs[i].append(dist.log_prob(torch.tensor(action)).unsqueeze(0))
                episode_values[i].append(val_seeker)

            next_obs, rewards, dones, info = env.step(actions)
            for i in range(num_agents):
                episode_rewards[i].append(rewards[i])

            total_episode_reward += sum(rewards)
            obs = next_obs

            if all(dones) or getattr(world, 'game_over', False):
                break

            #update policy and critic
            seeker_log_probs = []
            for i in range(num_agents):
                if len(episode_rewards[i]) == 0:
                    continue

                returns, advantages = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)

                for log_p, adv in zip(episode_log_probs[i], advantages):
                    policy_loss = -log_p * adv.item()
                    seeker_log_probs.append(policy_loss)

            if len(seeker_log_probs) > 0:
                seeker_actor_optimizer.zero_grad()
                seeker_policy_loss = torch.stack(seeker_log_probs).sum()
                seeker_policy_loss.backward()
                seeker_actor_optimizer.step()

            values, returns = [], []
            for i in range(num_agents):
                if len(episode_rewards[i]) == 0:
                    continue

                ret, _ = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
                returns.append(ret)
                values.append(torch.cat(episode_values[i]).squeeze())

            if(len(values) > 0):
                val_tensor = torch.cat(values)
                ret_tensor = torch.cat(returns)
                critic_loss = nn.MSELoss()(val_tensor, ret_tensor)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Total Reward: {total_episode_reward}")

    return seeker_actor

def train_with_static_guards(trained_seeker_actor=None):
    env, world = make_train_env(stage=2)
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    num_agents = len(env.agents)
    global_state_dim = obs_dim*num_agents

    if trained_seeker_actor is None:
        seeker_actor = Actor(obs_dim, action_dim)
    else:
        seeker_actor = trained_seeker_actor
    critic = Critic(global_state_dim)

    seeker_actor_optimizer = optim.Adam(seeker_actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99

    for episode in range(1000):
        obs = env.reset()
        episode_rewards = [[] for _ in range(num_agents)]
        episode_log_probs = [[] for _ in range(num_agents)]
        episode_values = [[] for _ in range(num_agents)]
        total_episode_reward = 0

        for _ in range(100):
            global_state = torch.FloatTensor(np.concatenate(obs)).unsqueeze(0)
            val_seeker = critic(global_state)

            actions = []
            for i, agent in enumerate(env.agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                
                #how are guards supposed to be static??
                if agent.role == "guard":
                    continue  # Guards are static, no action selection

                dist = seeker_actor(obs_tensor)
                action = dist.sample().item()
                actions.append(action)

                episode_log_probs[i].append(dist.log_prob(torch.tensor(action)).unsqueeze(0))
                episode_values[i].append(val_seeker)

            next_obs, rewards, dones, info = env.step(actions)
            for i in range(num_agents):
                episode_rewards[i].append(rewards[i])

            total_episode_reward += sum(rewards)
            obs = next_obs

            if all(dones) or getattr(world, 'game_over', False):
                break

        seeker_log_probs = []
        for i in range(num_agents):
            if env.agents[i].role == "guard":
                continue  # Skip guards for policy updates
            _, advantages = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            for log_p, adv in zip(episode_log_probs[i], advantages):
                policy_loss = -log_p * adv.item()
                seeker_log_probs.append(policy_loss)

        if len(seeker_log_probs) > 0:
            seeker_actor_optimizer.zero_grad()
            seeker_policy_loss = torch.stack(seeker_log_probs).sum()
            seeker_policy_loss.backward()
            seeker_actor_optimizer.step()

        values, returns = [], []
        for i in range(num_agents):
            if len(episode_rewards[i]) == 0:
                continue

            ret, _ = returns_and_advantages(episode_rewards[i], episode_values[i], gamma)
            returns.append(ret)
            values.append(torch.cat(episode_values[i]).squeeze())

        if(len(values) > 0):
            val_tensor = torch.cat(values)
            ret_tensor = torch.cat(returns)
            critic_loss = nn.MSELoss()(val_tensor, ret_tensor)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_episode_reward}")

#train against scripted baseline opponents

#train with self-play (both seekers and guards are learning agents)

def main():
    print("Train seekers with no opponents")
    print("Train seekers with static guards")
    print("Train seekers with scripted baseline opponents")
    print("Train seekers and guards with self-play")

if __name__ == "__main__":
    main()
        

