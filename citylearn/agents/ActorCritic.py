from typing import Any, List
import numpy as np
import torch
import torch.optim as optim

from citylearn.agents.rbc import OptimizedRBC
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from citylearn.rl import Actor, Critic, ReplayBuffer1  # 

class AC(RLC):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        """Simplified Actor-Critic (AC) RL agent for CityLearn."""
        super().__init__(env, **kwargs)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer_capacity = getattr(self, "replay_buffer_capacity", 100000)
        self.replay_buffer = [ReplayBuffer1(int(self.replay_buffer_capacity)) for _ in self.action_space]

        # Initialize actor-critic networks
        self.actor = [None for _ in self.action_space]
        self.critic = [None for _ in self.action_space]
        self.actor_optimizer = [None for _ in self.action_space]
        self.critic_optimizer = [None for _ in self.action_space]

        self.set_networks()

    def set_networks(self):
        """Initialize actor-critic networks for each action space."""
        for i in range(len(self.action_dimension)):
            state_dim = self.observation_dimension[i]
            action_dim = self.action_dimension[i]
            print(f"ActorCritic - State Dim: {state_dim}, Action Dim: {action_dim}")


            self.actor[i] = Actor(state_dim, action_dim, seed=42).to(self.device)  # Use Actor from RL file
            self.critic[i] = Critic(state_dim, action_dim, seed=42).to(self.device) # Use Critic from RL file

            self.actor_optimizer[i] = optim.Adam(self.actor[i].parameters(), lr=self.lr)
            self.critic_optimizer[i] = optim.Adam(self.critic[i].parameters(), lr=self.lr)

    def update(self, observations, actions, reward, next_observations, done):
        """Train the Actor-Critic network with replay buffer experience."""
        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            o = torch.FloatTensor(o).to(self.device)
            n = torch.FloatTensor(n).to(self.device)
            a = torch.FloatTensor(a).to(self.device)
            r = torch.FloatTensor([r]).to(self.device)
            d = torch.FloatTensor([done]).to(self.device)

            # Compute value estimates
            value = self.critic[i](o, a)
            next_value = self.critic[i](n, a).detach()

            # Compute target value
            target_value = r + (1 - d) * self.discount * next_value

            # Compute critic loss (Mean Squared Error)
            critic_loss = torch.nn.MSELoss()(value, target_value)

            # Update Critic
            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[i].step()

            # Compute policy loss (advantage function)
            advantage = (target_value - value).detach()
            policy_loss = -(advantage.mean() * self.actor[i](o).sum())

            # Update Actor
            self.actor_optimizer[i].zero_grad()
            policy_loss.backward()
            self.actor_optimizer[i].step()

    #def predict(self, observations, deterministic=False):
        #"""Select an action based on current policy."""
     #   actions = []
      #  for i, o in enumerate(observations):
       #     o = torch.FloatTensor(o).unsqueeze(0).to(self.device)
        #    a = self.actor[i](o)
         #   actions.append(a.detach().cpu().numpy()[0])
        #return actions

    def predict(self, observations, deterministic=False):
            actions_return = None
            if self.time_step > self.end_exploration_time_step or deterministic:
                if deterministic:
                    actions_return = self.get_deterministic_actions(observations)
                else:
                    actions_return = self.get_exploration_prediction(observations)
            else:
                actions_return = self.get_exploration_prediction(observations)
    
            data_to_append = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
    
            #print(data_to_append)
            #print(type(data_to_append))
            # Append the data to the file
            with open('method_calls.pkl', 'ab') as f:
                pickle.dump(data_to_append, f)
    
            self.next_time_step()
            return actions_return
    def predict_deterministic(self, encoded_observations):
        actions_return = None
        with torch.no_grad():
            actions_return = [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                    for actor, obs in zip(self.actors, encoded_observations)]
        return actions_return

class ACOptimizedRBC(AC):
    """Uses OptimizedRBC for initial exploration before switching to Actor-Critic RL."""

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.rbc = OptimizedRBC(env, **kwargs)  # Rule-Based Controller

    def get_exploration_prediction(self, states: List[float]) -> List[float]:
        """Use rule-based control (RBC) during exploration phase."""
        return self.rbc.predict(states)

    def predict(self, observations: List[List[float]], deterministic: bool = None):
        """Hybrid action selection: RBC first, AC later."""
        deterministic = False if deterministic is None else deterministic

        if self.time_step > self.end_exploration_time_step or deterministic:
            actions = super().predict(observations, deterministic)
        else:
            actions = self.get_exploration_prediction(observations)

        self.actions = actions
        self.next_time_step()
        return actions
