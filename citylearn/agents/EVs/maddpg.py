import torch
import torch.nn.functional as F
from citylearn.agents.rlc import RLC
import numpy as np
from citylearn.rl import Actor, Critic, OUNoise

class MADDPG(RLC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.actors = [Actor(self.observation_dimension[i], self.action_dimension[i], self.action_space[i]).to(self.device) for i in range(len(self.action_space))]
        self.critics = [Critic(self.observation_dimension[i], self.action_dimension[i]).to(self.device) for i in range(len(self.action_space))]

        self.actors_target = [Actor(self.observation_dimension[i], self.action_dimension[i], self.action_space[i]).to(self.device) for i in range(len(self.action_space))]
        self.critics_target = [Critic(self.observation_dimension[i], self.action_dimension[i]).to(self.device) for i in range(len(self.action_space))]

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=self.lr) for i in range(len(self.action_space))]
        self.critics_optimizer = [torch.optim.Adam(self.critics[i].parameters(), lr=self.lr) for i in range(len(self.action_space))]

        self.noise = [OUNoise(self.action_dimension[i], scale=1.0 ) for i in range(len(self.action_space))]

    def update(self, observations, actions, reward, next_observations, done):
        # For each agent
        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            o = self.get_encoded_observations(i, o)
            n = self.get_encoded_observations(i, n)

            if self.normalized[i]:
                o = self.get_normalized_observations(i, o)
                n = self.get_normalized_observations(i, n)
                r = self.get_normalized_reward(i, r)
            else:
                pass

            self.replay_buffer[i].push(o, a, r, n, done)

            if self.time_step >= self.start_training_time_step and self.batch_size <= len(self.replay_buffer[i]):
                # Update policy and value parameters using given batch of experience tuples.
                experiences = self.replay_buffer[i].sample(self.batch_size)
                self.learn(experiences, i)

    def learn(self, experiences, i):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = [self.actors_target[i](next_states[:, i, :]) for i in range(self.num_agents)]
        actions_next = torch.cat(actions_next, dim=1)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (i * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # Gradient clipping
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [self.actor_local[i](states[:, i, :]) for i in range(self.num_agents)]
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        for i in range(self.num_agents):
            self.soft_update(self.actor_local[i], self.actor_target[i], self.tau)

    def predict(self, observations, deterministic=False):
        """Decides an action for each agent given their observations.

        Args:
            observations (List[List[float]]): A list of observations for each agent.
            deterministic (bool, optional): Whether to use deterministic policy.
                Defaults to False.

        Returns:
            List[List[float]]: A list of actions for each agent.
        """
        actions = []

        # Loop over each agent
        for i, agent in enumerate(self.agents):
            # Reshape observation for the agent
            obs = np.array(observations[i]).reshape(1, -1)

            if self.time_step > self.end_exploration_time_step or deterministic:
                # If we are past the exploration phase, use the actor network to decide action
                agent.actor_local.eval()
                with torch.no_grad():
                    action = agent.actor_local(obs).detach().cpu().numpy()
                agent.actor_local.train()
            else:
                # During the exploration phase, choose action randomly
                action = agent.action_space.sample()

            actions.append(action)

        self.next_time_step()

        return actions

    def soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
