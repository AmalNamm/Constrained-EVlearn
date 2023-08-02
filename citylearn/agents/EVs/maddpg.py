from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from citylearn.rl import Actor, Critic, OUNoise, ReplayBuffer1
import random
import numpy.typing as npt

class MADDPG(RLC):
    def __init__(self, env: CityLearnEnv, actor_units: tuple = (256, 128), critic_units: tuple = (256, 128), buffer_size: int = int(1e6), batch_size: int = 64, gamma: float = 0.99, *args, **kwargs):

        super().__init__(env, **kwargs)

        # Retrieve number of agents
        self.num_agents = len(self.action_space)

        # Discount factor for the MDP
        self.gamma = gamma

        # Replay buffer and batch size
        self.replay_buffer = ReplayBuffer1(capacity=buffer_size, num_agents=self.num_agents)
        self.batch_size = batch_size

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = random.randint(0, 100_000_000)
        self.actor_units = actor_units
        self.critic_units = critic_units

        self.actors = [
            Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, *self.actor_units).to(
                self.device) for i in range(len(self.action_space))] #basicly for each building it creates an actor network
        self.critics = [
            Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, *self.critic_units).to(
                self.device) for _ in range(len(self.action_space))]

        self.actors_target = [
            Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, *self.actor_units).to(
                self.device) for i in range(len(self.action_space))]
        self.critics_target = [
            Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, *self.critic_units).to(
                self.device) for _ in range(len(self.action_space))]

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=self.lr) for i in
                                 range(len(self.action_space))]
        self.critics_optimizer = [torch.optim.Adam(self.critics[i].parameters(), lr=self.lr) for i in
                                  range(len(self.action_space))]

        self.noise = [OUNoise(self.action_space[i].shape[0], self.seed) for i in range(len(self.action_space))]

    def update(self, observations, actions, reward, next_observations, done):
        self.replay_buffer.push(observations, actions, reward, next_observations, done)

        if len(self.replay_buffer) < self.batch_size:
            return

        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.replay_buffer.sample(
            self.batch_size)

        obs_tensors = []
        next_obs_tensors = []
        actions_tensors = []

        for agent_num in range(len(self.action_space)):
            obs_tensors.append(
                torch.stack([torch.FloatTensor(self.get_encoded_observations(agent_num, obs)).to(self.device)
                             for obs in obs_batch[agent_num]]))
            next_obs_tensors.append(
                torch.stack([torch.FloatTensor(self.get_encoded_observations(agent_num, next_obs)).to(self.device)
                             for next_obs in next_obs_batch[agent_num]]))
            actions_tensors.append(
                torch.stack([torch.FloatTensor(action).to(self.device)
                             for action in actions_batch[agent_num]]))

        reward_tensors = torch.from_numpy(np.array(rewards_batch, dtype=np.float32)).to(self.device)

        for agent_num, (actor, critic, actor_target, critic_target, actor_optim, critic_optim) in enumerate(
                zip(self.actors, self.critics, self.actors_target, self.critics_target, self.actors_optimizer,
                    self.critics_optimizer)):
            obs_full = torch.cat(obs_tensors, dim=1)
            next_obs_full = torch.cat(next_obs_tensors, dim=1)
            action_full = torch.cat(actions_tensors, dim=1)

            # Update critic
            Q_expected = critic(obs_full, action_full)
            next_actions = [self.actors_target[i](next_obs_tensors[i]) for i in range(self.num_agents)]
            next_actions_full = torch.cat(next_actions, dim=1)
            Q_targets_next = critic_target(next_obs_full, next_actions_full)
            dones_tensors = torch.tensor(dones_batch[agent_num]).unsqueeze(1).float().to(self.device)
            Q_targets = reward_tensors.unsqueeze(1) + (self.gamma * Q_targets_next * (1 - dones_tensors))
            critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # Update actor
            predicted_actions = [self.actors[i](obs_tensors[i]) for i in range(self.num_agents)]
            predicted_actions_full = torch.cat(predicted_actions, dim=1)
            actor_loss = -critic(obs_full, predicted_actions_full).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # Update target networks
            self.soft_update(critic, critic_target)
            self.soft_update(actor, actor_target)

    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    #def update(self, observations, actions, reward, next_observations, done):
#
    #    obs_tensors = [torch.FloatTensor(self.get_encoded_observations(i, obs)).to(self.device) for i, obs in
    #                   enumerate(observations)]
    #    next_obs_tensors = [torch.FloatTensor(self.get_encoded_observations(i, next_obs)).to(self.device) for
    #                        i, next_obs in enumerate(next_observations)]
##
    #    actions_tensors = [torch.FloatTensor(act).to(self.device) for act in actions]
    #    reward_tensor = torch.FloatTensor(reward).to(self.device)
    #    done_tensor = torch.tensor([done], dtype=torch.bool).to(self.device)
#
    #    for agent_num, (actor, critic, actor_target, critic_target, actor_optim, critic_optim) in enumerate(
    #            zip(self.actors, self.critics, self.actors_target, self.critics_target, self.actors_optimizer,
    #                self.critics_optimizer)):
    #        actor_optim.zero_grad()
    #        pred_actions = actor(obs_tensors[agent_num])
    #        all_obs = torch.cat(obs_tensors, dim=-1).view(1, -1)
    #        all_actions = torch.cat(actions_tensors, dim=-1).view(1, -1)  # Concatenate all actions
##
    #        pred_actions = pred_actions.unsqueeze(0)  # add an extra dimension
    #        actions_other_agents = torch.cat(
    #            [actions_tensors[i] for i in range(len(actions_tensors)) if i != agent_num], dim=-1).unsqueeze(0)
##
    #        actor_loss = -critic(all_obs, torch.cat([pred_actions, actions_other_agents], dim=-1)).mean()
    #        actor_loss.backward()
    #        actor_optim.step()
##
    #        critic_optim.zero_grad()
    #        all_next_obs = torch.cat(next_obs_tensors, dim=-1).view(1, -1)
    #        target_actions = actor_target(next_obs_tensors[agent_num]).unsqueeze(0)
##
    #        target_actions_other_agents = torch.cat(
    #            [self.actors_target[i](next_obs_tensors[i]) for i in range(len(next_obs_tensors)) if i != agent_num],
    #            dim=-1).unsqueeze(0)
    #        all_target_actions = torch.cat([target_actions, target_actions_other_agents],
    #                                       dim=-1).detach()  # Concatenate all target actions
##
    #        next_Q = critic_target(all_next_obs, all_target_actions)
    #        expected_Q = reward_tensor[agent_num] + self.discount * next_Q * (~done_tensor)
    #        current_Q = critic(all_obs, all_actions)
    #        critic_loss = F.mse_loss(current_Q, expected_Q.detach())
    #        critic_loss.backward()
    #        critic_optim.step()
##
    #        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
    #            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
##
    #        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
    #            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def predict(self, observations, deterministic=False):
        with torch.no_grad():
            encoded_observations = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
            if deterministic:
                return [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                        for actor, obs in zip(self.actors, encoded_observations)]
            else:
                return [self.noise[i].sample() + action
                        for i, action in enumerate(self.predict(observations, True))]

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(self.encoders[index]*np.array(observations, dtype=float)) if j != None], dtype = float)
