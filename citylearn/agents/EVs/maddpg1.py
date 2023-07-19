import numpy as np
import torch
import torch.nn.functional as F
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from citylearn.rl import Actor, Critic, OUNoise
from citylearn.rl import ReplayBuffer1 as ReplayBuffer
import random
import numpy.typing as npt


class MADDPG(RLC):
    def __init__(self, env: CityLearnEnv, actor_units: tuple = (256, 128), critic_units: tuple = (256, 128), *args, **kwargs):

        super().__init__(env, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = random.randint(0, 100_000_000)
        self.actor_units = actor_units
        self.critic_units = critic_units

        self.actors = [
            Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, *self.actor_units).to(
                self.device) for i in range(len(self.action_space))]
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

        self.batch_size = 64
        self.buffer_size = int(1e6)
        self.memory = ReplayBuffer(action_size=sum(self.action_dimension), buffer_size=self.buffer_size, batch_size=self.batch_size, seed=self.seed, device=self.device)

    def update(self, observations, actions, reward, next_observations, done):
        # Save experience in replay memory
        for i in range(len(self.action_space)):
            self.memory.add(observations[i], actions[i], reward[i], next_observations[i], done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn1(experiences)

    def learn1(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        for agent_num, (actor, critic, actor_target, critic_target, actor_optim, critic_optim) in enumerate(
                zip(self.actors, self.critics, self.actors_target, self.critics_target, self.actors_optimizer,
                    self.critics_optimizer)):
            actor_optim.zero_grad()
            pred_actions = actor(states)
            all_obs = torch.cat(states, dim=-1).view(1, -1)
            all_actions = torch.cat(actions, dim=-1).view(1, -1)

            pred_actions = pred_actions.unsqueeze(0)
            actions_other_agents = torch.cat(
                [actions[i] for i in range(len(actions)) if i != agent_num], dim=-1).unsqueeze(0)

            actor_loss = -critic(all_obs, torch.cat([pred_actions, actions_other_agents], dim=-1)).mean()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            all_next_obs = torch.cat(next_states, dim=-1).view(1, -1)
            target_actions = actor_target(next_states).unsqueeze(0)

            target_actions_other_agents = torch.cat(
                [self.actors_target[i](next_states[i]) for i in range(len(next_states)) if i != agent_num],
                dim=-1).unsqueeze(0)
            all_target_actions = torch.cat([target_actions, target_actions_other_agents],
                                           dim=-1).detach()

            next_Q = critic_target(all_next_obs, all_target_actions)
            expected_Q = rewards + self.discount * next_Q * (~dones)
            current_Q = critic(all_obs, all_actions)
            critic_loss = F.mse_loss(current_Q, expected_Q.detach())
            critic_loss.backward()
            critic_optim.step()

            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def predict(self, observations, deterministic=False):
        """Return actions for given state as per current policy."""
        with torch.no_grad():
            if deterministic:
                return [actor(torch.FloatTensor(observation).to(self.device)).cpu().numpy()
                        for actor, observation in zip(self.actors, observations)]
            else:
                return [self.noise[i].get_action(action, self.time_step)
                        for i, action in enumerate(self.predict(observations, True))]
