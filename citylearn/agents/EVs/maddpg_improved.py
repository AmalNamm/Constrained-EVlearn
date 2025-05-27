from typing import Any, List, Optional, Union
from citylearn.preprocessing import Encoder, NoNormalization, PeriodicNormalization, OnehotEncoding ,RemoveFeature, Normalize
import numpy as np
import torch
import torch.nn.functional as F
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
#from citylearn.rl import Actor, Critic, OUNoise, ReplayBuffer2
from citylearn.rl_tgelu import Actor, Critic, ConstraintCritic, ReplayBuffer2
import random
import numpy.typing as npt
import timeit
from torch.cuda.amp import autocast, GradScaler
from citylearn.agents.rbc import RBC, BasicBatteryRBC, BasicRBC, V2GRBC, OptimizedRBC
from torch.utils.tensorboard.writer import SummaryWriter
import pickle
import time

import csv
import os

class MADDPG(RLC):
    def __init__(self, env: CityLearnEnv, actor_units: list = [256, 128], critic_units: list = [256, 128],
                 constraint_critic_units: Optional[list] = None, buffer_size: int = int(1e5), batch_size: int = 1024, 
                 gamma: float = 0.99, sigma: float = 0.2, target_update_interval: int = 2, lr_actor: float = 1e-5, 
                 lr_critic: float = 1e-4, lr_constraint_critic: Optional[float] = None, lr_dual: float = 1e-5, 
                 steps_between_training_updates: int = 5, decay_percentage: float = 0.995, tau: float = 1e-3,
                 target_network: bool = False, *args: Any, **kwargs: Any): ###NEW ### Added lr_dual: float = 1e-5

        self.target_network = target_network
        super().__init__(env, **kwargs)

        # Retrieve number of agents
        self.num_agents = len(self.action_space)

        # Discount factor for the MDP
        self.gamma = gamma

        # Replay buffer and batch size
        self.replay_buffer = ReplayBuffer2(capacity=buffer_size, num_agents=self.num_agents)
        self.batch_size = batch_size

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.seed = random.randint(0, 100_000_000)
        self.actor_units = actor_units
        self.critic_units = critic_units
        
        # If constraint critic units not provided, use the same as the regular critic
        self.constraint_critic_units = critic_units.copy() if constraint_critic_units is None else constraint_critic_units
        
        # If constraint critic learning rate not provided, use the same as the regular critic
        self.lr_constraint_critic = lr_critic if lr_constraint_critic is None else lr_constraint_critic
        
        self.tau = tau
        self.sigma = sigma

        ##added 06.05.2025 
        #set up a CSV log file
        self.lagrangian_logfile = "lagrangian_monitoring_New_Cost_inequality_Layer_Update_simpleEV resward.csv"
        if not os.path.exists(self.lagrangian_logfile):
            with open(self.lagrangian_logfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "agent", "lagrangian", "mean_constraint_violation"])

        # For monitoring reward vs constraint scales
        self.scale_monitor_callback = None

        # Initialize actors and critics
        # Each actor network handles its agent's local observation and action space, learning its own policy.
        self.actors = [
            Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, actor_units, tgelu_range=None, gamma=self.gamma).to(
                self.device) for i in range(len(self.action_space))
        ]  ##decentrailised Actor 20.02.2025
        
        #Each critic network, receiving global information, can assess the overall quality of the joint actions taken by all agents, which is particularly useful in cooperative or competitive multi-agent settings.
        #update Critic to QCritic
        self.critics = [
            Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_units).to(
                self.device) for _ in range(len(self.action_space))
        ] ##centrailised Critic 20.02.2025

        # Initialize target networks if target_network is True
        if self.target_network:
            self.actors_target = [
                Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, actor_units, tgelu_range=None, gamma=self.gamma).to(
                    self.device) for i in range(len(self.action_space))
            ] #updated to include gamma 
            self.critics_target = [
                Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_units).to(
                    self.device) for _ in range(len(self.action_space))
            ]

        self.actors_optimizer = [torch.optim.SGD(self.actors[i].parameters(), lr=lr_actor) for i in
                                 range(len(self.action_space))]
        self.critics_optimizer = [torch.optim.SGD(self.critics[i].parameters(), lr=lr_critic) for i in
                                  range(len(self.action_space))]

        ### NEW ### Constraint Critic and its target networks - now using specialized ConstraintCritic class ***
        #***
        #Using specialized ConstraintCritic that's optimized for constraint estimation
        self.constraint_critics = [
            ConstraintCritic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, self.constraint_critic_units, tgelu_range=None, gamma=self.gamma).to(
                self.device) for _ in range(len(self.action_space))
        ]

        if self.target_network:
            self.constraint_critics_target = [
                ConstraintCritic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, self.constraint_critic_units, tgelu_range=None, gamma=self.gamma).to(
                    self.device) for _ in range(len(self.action_space))
            ]

        # Possibly using a different learning rate for constraint critics
        self.constraint_critics_optimizer = [torch.optim.SGD(self.constraint_critics[i].parameters(), lr=self.lr_constraint_critic) for i in
                                  range(len(self.action_space))]

        #***

        ### NEW ### Lagrangian parameter and its optimizer ***
        #***
        #self.lagrangian = torch.tensor(1.0, requires_grad = True, device=self.device)
        #self.lambda_optimizer = torch.optim.SGD([self.lagrangian], lr=lr_dual)
        #***  ###To revisit 

        #Updating to Per-Agent Lagrange Multipliers Lambda_i 05.05.2025
        self.lagrangians = torch.ones(self.num_agents, device=self.device)
        self.lr_dual = lr_dual

        decay_factor = decay_percentage ** (1/self.env.time_steps)
        #self.noise = [OUNoise(self.action_space[i].shape[0], self.seed, sigma=sigma, decay_factor=decay_factor) for i in range(len(self.action_space))]

        self.target_update_interval = target_update_interval
        self.steps_between_training_updates = steps_between_training_updates
        self.scaler = GradScaler()
        self.exploration_done = False
        
        print(f"Initialized MADDPG with specialized critics:")
        print(f"  - Reward critics: {critic_units}, lr={lr_critic}")
        print(f"  - Constraint critics: {self.constraint_critic_units}, lr={self.lr_constraint_critic}")

    @classmethod
    def from_saved_model(cls, filename):
        """Initialize an agent from a saved model file."""

        # Load the saved data
        data = torch.load(filename)

        # Create an empty agent instance without calling the actual __init__ method
        agent = cls.__new__(cls)

        # Set up the agent's basic attributes using the loaded data
        observation_dimension = data['observation_dimension']
        action_dimension = data['action_dimension']
        agent.num_agents = data['num_agents']
        agent.seed = data['seed']
        agent.actor_units = data['actor_units']
        agent.critic_units = data['critic_units']
        agent.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actors and critics with loaded data
        agent.actors = [
            Actor(observation_dimension[i], action_dimension[i], agent.seed, agent.actor_units).to(
                agent.device)
            for i in range(agent.num_agents)
        ]
        agent.critics = [
            Critic(data['total_observation_dimension'], data['total_action_dimension'], agent.seed,
                   agent.critic_units).to(agent.device)
            for _ in range(agent.num_agents)
        ]

        # Initialize actors_target and critics_target with loaded data
        agent.actors_target = [
            Actor(observation_dimension[i], action_dimension[i], agent.seed, agent.actor_units).to(
                agent.device)
            for i in range(agent.num_agents)
        ]
        agent.critics_target = [
            Critic(data['total_observation_dimension'], data['total_action_dimension'], agent.seed,
                   agent.critic_units).to(agent.device)
            for _ in range(agent.num_agents)
        ]

        # Load the state dictionaries into the actor and critic models
        for actor, state_dict in zip(agent.actors, data['actors']):
            actor.load_state_dict(state_dict)
        for critic, state_dict in zip(agent.critics, data['critics']):
            critic.load_state_dict(state_dict)

        # Load the state dictionaries into the actor_target and critic_target models
        for actor_target, state_dict in zip(agent.actors_target, data['actors_target']):
            actor_target.load_state_dict(state_dict)
        for critic_target, state_dict in zip(agent.critics_target, data['critics_target']):
            critic_target.load_state_dict(state_dict)

        # If you've saved optimizers' states, you can initialize and load them similarly (optional)
        agent.actors_optimizer = [torch.optim.SGD(actor.parameters()) for actor in agent.actors]
        agent.critics_optimizer = [torch.optim.SGD(critic.parameters()) for critic in agent.critics]
        for optimizer, state_dict in zip(agent.actors_optimizer, data.get('actors_optimizer', [])):
            optimizer.load_state_dict(state_dict)
        for optimizer, state_dict in zip(agent.critics_optimizer, data.get('critics_optimizer', [])):
            optimizer.load_state_dict(state_dict)

        return agent
 
    ### NEW ### 
    #***
   # def compute_constraint_cost(self, agent_num, actions): #action space
    #    """
    #    Computes the constraint cost for a given agent based on EV charging power limits.
    #    Assumes that each action component corresponds to a charger control for the building (agent).
    #    Returns a tensor of shape (batch_size, 1) with the sum of violation counts for each sample.
    #    """
    #    cost = [] #initialized to store the violation count for each sample in the batch.
    #    building = self.env.buildings[agent_num]
    #    # actions is a tensor of shape (batch_size, action_dim)
    #    for action in actions:  # iterate over batch dimension
     #       sample_cost = 0.0
     #       if building.chargers:
     #           # Assume order of action elements corresponds to the order of chargers
     #           for j, charger in enumerate(building.chargers):
     #               act_value = action[j].item()  # extract scalar action for charger j #double check
     #               real_power = act_value * charger.nominal_power  # scale by nominal power
     #               if real_power > charger.max_charging_power or real_power < charger.min_charging_power:
     #                   sample_cost += 1.0  # count one violation per charger violation
     #       cost.append(sample_cost)
     #   return torch.tensor(cost, dtype=torch.float32, device=self.device).view(-1, 1)
    #***

    # === Updated constraint cost function with inequality constraint and tolerance ===

    
    ### New -- Added 12.05.2025
    # === Updated get_constraint_cost: smooth, normalized inequality-based version ===

    def get_constraint_cost(self, observations, actions, next_observations):
        """
        Computes smoothed, inequality-based constraint cost with tolerance for EV charging.
        Returns a normalized cost per sample (batch_size, 1).
        """
        #The final output is:
        #A tensor of shape (num_buildings, 1)
        #Each element represents the total constraint violation for one building
        costs = []
        tolerance = 0.1
    
        for agent_num, (obs, action, next_obs) in enumerate(zip(observations, actions, next_observations)):
            total_cost = 0.0
            building = self.env.buildings[agent_num]
    
            if building.chargers:
                for j, charger in enumerate(building.chargers):
                    real_power = action[j] * charger.nominal_power
    
                    over = max(0.0, real_power - charger.max_charging_power - tolerance)
                    under = max(0.0, charger.min_charging_power - real_power - tolerance)
                    violation_cost = over**2 + under**2

                    #Normalize violations relative to allowed power range
                    #Purpose: Makes constraints comparable across different chargers with different power ranges
    
                    range_width = charger.max_charging_power - charger.min_charging_power
                    if range_width > 0:
                        violation_cost /= (range_width**2)
    
                    total_cost += violation_cost
    
            costs.append([total_cost])
    
        return torch.tensor(costs, dtype=torch.float32, device=self.device).view(-1, 1)

    

    # We do not need all of these values now. Just future proofing. 
    def get_constraint_cost_V1(self, observations, actions, next_observations):
        costs = [] 
        for agent_num, (obs, action, next_obs) in enumerate(zip(observations, actions, next_observations)): # Enumerate the agents
            sample_cost = 0.0
            #cost = [] not used ?
            building = self.env.buildings[agent_num]
            if building.chargers:
                for j, charger in enumerate(building.chargers): # Enumerate the chargers for the specific agent
                    act_value = action[j]
                    real_power = act_value * charger.nominal_power
                    if real_power > charger.max_charging_power or real_power < charger.min_charging_power:
                        sample_cost += 1
            costs.append([sample_cost])
        #return costs
        #Add Device Handling
        return torch.tensor(costs, dtype=torch.float32, device=self.device).view(-1, 1)
                
                
    #def update(self, observations, actions, reward, next_observations, constraint_value, done): #ADD Constraint
        #self.replay_buffer.push(observations, actions, reward, next_observations,constraint_value, done) #updated
    def update(self, observations, actions, reward, next_observations, done): #ADD Constraint
        # Pre-encode observations
        # CPU Operations (slow)
        encoded_obs = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
        encoded_next_obs = [self.get_encoded_observations(i, next_obs) for i, next_obs in enumerate(next_observations)]
    
        # Calculate constraint costs for current transition
        cons = self.get_constraint_cost(observations, actions, next_observations)
        ## Store in replay buffer with other experience data
        self.replay_buffer.push(encoded_obs, actions, reward, encoded_next_obs, cons, done)

        if len(self.replay_buffer) < self.batch_size:
            return None, None, None

        if not self.exploration_done:
            if self.time_step < self.end_exploration_time_step:
                return None, None, None
            elif self.time_step == self.end_exploration_time_step:
                self.exploration_done = True
                print("Ended exploration")
                return None, None, None
                
        if self.time_step % self.steps_between_training_updates != 0:
            return None, None, None

        # Sample a batch of experiences from the replay buffer
        obs_batch, actions_batch, rewards_batch, next_obs_batch, constraint_batch, dones_batch = self.replay_buffer.sample(
            self.batch_size)

        # Convert to tensors and prepare for GPU processing
        obs_tensors = []
        next_obs_tensors = []
        actions_tensors = []
        reward_tensors = []
        constraint_tensor = []
        dones_tensors = []

        for agent_num in range(len(self.action_space)):
            # Moving to GPU (slow transfer)
            obs_tensors.append(
                torch.stack([torch.FloatTensor(obs).to(self.device)
                            for obs in obs_batch[agent_num]]))
            next_obs_tensors.append(
                torch.stack([torch.FloatTensor(next_obs).to(self.device)
                            for next_obs in next_obs_batch[agent_num]]))
            actions_tensors.append(
                torch.stack([torch.FloatTensor(action).to(self.device)
                            for action in actions_batch[agent_num]]))
            reward_tensors.append(
                torch.tensor(rewards_batch[agent_num], dtype=torch.float32).to(self.device).view(-1, 1))
            constraint_tensor.append(
                torch.stack([torch.FloatTensor(cons).to(self.device) 
                            for cons in constraint_batch[agent_num]]))
            dones_tensors.append(
                torch.tensor(dones_batch[agent_num], dtype=torch.float32).to(self.device).view(-1, 1))

        # Concatenate observations and actions for centralized critics
        obs_full = torch.cat(obs_tensors, dim=1)
        next_obs_full = torch.cat(next_obs_tensors, dim=1)
        action_full = torch.cat(actions_tensors, dim=1)

        # Track metrics for all agents
        global_constraint_costs = []
        constraint_losses = []
        critic_losses = []
        actor_losses = []

        # If not using target networks, use main networks as targets
        if not self.target_network:
            self.actors_target = self.actors
            self.critics_target = self.critics
            self.constraint_critics_target = self.constraint_critics

        # Update networks for each agent
        for agent_num, (actor, critic, constraint_critic, actor_target, critic_target, constraint_critic_target, 
                        actor_optim, critic_optim, constraint_optim) in enumerate(
                zip(self.actors, self.critics, self.constraint_critics, 
                    self.actors_target, self.critics_target, self.constraint_critics_target,
                    self.actors_optimizer, self.critics_optimizer, self.constraint_critics_optimizer)):
            
            # 1. Update Reward Critic (Q-value estimator)
            with autocast():
                # Current state-action value
                Q_expected = critic(obs_full, action_full)
                
                # Next state-action value with target networks
                next_actions = [self.actors_target[i](next_obs_tensors[i]) for i in range(self.num_agents)]
                next_actions_full = torch.cat(next_actions, dim=1)
                Q_targets_next = critic_target(next_obs_full, next_actions_full)
                
                # TD target with reward
                Q_targets = reward_tensors[agent_num] + (self.gamma * Q_targets_next * (1 - dones_tensors[agent_num]))
                
                # MSE loss for reward critic
                critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
                critic_losses.append(critic_loss.item())

            # Optimize reward critic
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(critic_optim)
            critic_optim.zero_grad()
            self.scaler.update()

            # 2. Update Constraint Critic (specialized for constraint estimation)
            with autocast():
                # Current state-action constraint value # Current predicted constraint value
                constraint_expected = constraint_critic(obs_full, action_full)
                
                # Actual constraint cost from experience
                constraint_cost = constraint_tensor[agent_num]
                
                # Next state-action constraint value with target networks
                next_actions = [self.actors_target[i](next_obs_tensors[i]) for i in range(self.num_agents)]
                next_actions_full = torch.cat(next_actions, dim=1)
                constraint_targets_next = constraint_critic_target(next_obs_full, next_actions_full)
                
                # TD target with constraint cost
                constraint_targets = constraint_cost + (self.gamma * constraint_targets_next * (1 - dones_tensors[agent_num]))
                
                # MSE loss for constraint critic
                constraint_loss = F.mse_loss(constraint_expected, constraint_targets.detach())
                constraint_losses.append(constraint_loss.item())

            # Optimize constraint critic
            self.scaler.scale(constraint_loss).backward()
            self.scaler.step(constraint_optim)
            constraint_optim.zero_grad()
            self.scaler.update()

            # Store constraint costs for Lagrangian updates
            global_constraint_costs.append(constraint_cost)

            # 3. Update Actor using both critics
            with autocast():
                # Get predicted actions from all actors
                predicted_actions = [self.actors[i](obs_tensors[i]) for i in range(self.num_agents)]
                predicted_actions_full = torch.cat(predicted_actions, dim=1)
                
                # Objective: Maximize reward (Q-value) while minimizing constraint violations
                # Weighted by Lagrangian multiplier
                reward_term = critic(obs_full, predicted_actions_full).mean()
                constraint_term = constraint_critic(obs_full, predicted_actions_full).mean()
                
                # Combined loss with Lagrangian weighting
                actor_loss = -reward_term + self.lagrangians[agent_num] * constraint_term
                actor_losses.append(actor_loss.item())

            # Optimize actor
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(actor_optim)
            actor_optim.zero_grad()
            self.scaler.update()

            # 4. Update target networks if using them
            if self.target_network and self.time_step % self.target_update_interval == 0:
                self.soft_update(critic, critic_target, self.tau)
                self.soft_update(actor, actor_target, self.tau)
                self.soft_update(constraint_critic, constraint_critic_target, self.tau)

        # 5. Update Lagrangian multipliers based on constraint violations
            ## Calculate average constraint violations across batch
        mean_constraint_violations = torch.stack([cons_tensor.mean() for cons_tensor in constraint_tensor])

        # Gradient-free update of Lagrangian multipliers
        with torch.no_grad():
            # Increase multiplier when constraint is violated, keep >= 0
            self.lagrangians += self.lr_dual * mean_constraint_violations
            self.lagrangians.clamp_(min=0.0)

        # Log Lagrangian updates to CSV
        for i in range(self.num_agents):
            with open(self.lagrangian_logfile, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.time_step,
                    i,
                    float(self.lagrangians[i].item()),
                    float(mean_constraint_violations[i].item())
                ])

        # Return loss values for monitoring
        return constraint_losses, critic_losses, self.lagrangians.clone().detach().cpu().numpy()


    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_deterministic_actions(self, observations):
        with torch.no_grad():
        # Pre-encode observations
            encoded_observations = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
            to_return = [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                         for actor, obs in zip(self.actors, encoded_observations)]
            return to_return


    def predict(self, observations, deterministic=False):
        # Pre-encode observations
        encoded_observations = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
        
        actions_return = None
        if self.time_step > self.end_exploration_time_step or deterministic:
            if deterministic:
                actions_return = self.predict_deterministic(encoded_observations)
            else:
                actions_return = self.get_exploration_prediction(observations)  # This will encode internally
        else:
            actions_return = self.get_exploration_prediction(observations)  # This will encode internally

        # Save encoded observations if needed
        #with open('method_calls.pkl', 'ab') as f:
        #    pickle.dump(encoded_observations, f)

        self.next_time_step()
        return actions_return

    

    def predict_deterministic(self, encoded_observations):
        actions_return = None
        with torch.no_grad():
            actions_return = [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                    for actor, obs in zip(self.actors, encoded_observations)]
        return actions_return

    def get_exploration_prediction(self, states: List[List[float]]) -> List[float]:
        # Pre-encode states for deterministic actions
        encoded_states = [self.get_encoded_observations(i, state) for i, state in enumerate(states)]
        deterministic_actions = self.predict_deterministic(encoded_states)

        # Generate random noise and print its sign for each action
        random_noises = []
        for action in deterministic_actions:
            bias = 0.3
            noise = np.random.normal(scale=self.sigma) - bias
            random_noises.append(noise)

        actions = [noise + action for action, noise in zip(deterministic_actions, random_noises)]
        clipped_actions = [np.clip(action, -1, 1) for action in actions]
        actions_return = [action.tolist() for action in clipped_actions]

        # Hard Constraints to exploration
        for i, b in enumerate(self.env.buildings):
            if b.chargers:
                for charger_index, charger in reversed(list(enumerate(b.chargers))):
                    # If no EV is connected, set action to 0
                    if not charger.connected_ev:
                        actions_return[i][-charger_index - 1] = 0.0001

        return actions_return

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(self.encoders[index]*np.array(observations, dtype=float)) if j != None], dtype = float)

    def reset(self):
        super().reset()

    def set_encoders(self) -> List[List[Encoder]]:
        r"""Get observation value transformers/encoders for use in MARLISA agent internal regression model.

        The encoder classes are defined in the `preprocessing.py` module and include `PeriodicNormalization` for cyclic observations,
        `OnehotEncoding` for categorical obeservations, `RemoveFeature` for non-applicable observations given available storage systems and devices
        and `Normalize` for observations with known minimum and maximum boundaries.

        Returns
        -------
        encoders : List[Encoder]
            Encoder classes for observations ordered with respect to `active_observations`.
        """

        encoders = []

        for o, s in zip(self.observation_names, self.observation_space):
            e = []

            remove_features = [
                'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',
                'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h',
                'outdoor_relative_humidity', 'outdoor_relative_humidity_predicted_6h',
                'outdoor_relative_humidity_predicted_12h', 'outdoor_relative_humidity_predicted_24h',
                'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h',
                'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h'
            ]

            for i, n in enumerate(o):
                if n in ['month', 'hour']:
                    e.append(PeriodicNormalization(s.high[i]))

                elif any(item in n for item in ["required_soc_departure", "estimated_soc_arrival", "ev_soc"]):
                    e.append(Normalize(s.low[i], s.high[i]))

                elif any(item in n for item in ["estimated_departure_time", "estimated_arrival_time"]):
                    e.append(OnehotEncoding([-1] + list(range(0, 25))))

                elif n in ['day_type']:
                    e.append(OnehotEncoding([1, 2, 3, 4, 5, 6, 7, 8]))

                elif n in ["daylight_savings_status"]:
                    e.append(OnehotEncoding([0, 1]))

                elif n in remove_features:
                    e.append(RemoveFeature())

                else:
                    e.append(NoNormalization())

            encoders.append(e)

        return encoders

    def save_maddpg_model(agent, filename):
        """Save the model's actor and critic networks along with other essential data."""
        data = {
            'actors': [actor.state_dict() for actor in agent.actors],
            'critics': [critic.state_dict() for critic in agent.critics],
            'actors_target': [actor_target.state_dict() for actor_target in agent.actors_target],
            'critics_target': [critic_target.state_dict() for critic_target in agent.critics_target],
            'actors_optimizer': [optimizer.state_dict() for optimizer in agent.actors_optimizer],
            'critics_optimizer': [optimizer.state_dict() for optimizer in agent.critics_optimizer],

            # Additional data for reinitializing the agent
            'observation_dimension': agent.observation_dimension,
            'action_dimension': agent.action_dimension,
            'num_agents': agent.num_agents,
            'seed': agent.seed,
            'actor_units': agent.actor_units,
            'critic_units': agent.critic_units,
            'device': agent.device.type,  # just save the type (e.g., 'cuda' or 'cpu')
            'total_observation_dimension': sum(agent.observation_dimension),
            'total_action_dimension': sum(agent.action_dimension)
        }
        torch.save(data, filename)

    def load_model(self, filename):
        """Load the model's actor and critic networks."""
        data = torch.load(filename)

        for actor, state_dict in zip(self.actors, data['actors']):
            actor.load_state_dict(state_dict)

        for critic, state_dict in zip(self.critics, data['critics']):
            critic.load_state_dict(state_dict)

        for actor_target, state_dict in zip(self.actors_target, data['actors_target']):
            actor_target.load_state_dict(state_dict)

        for critic_target, state_dict in zip(self.critics_target, data['critics_target']):
            critic_target.load_state_dict(state_dict)

        for optimizer, state_dict in zip(self.actors_optimizer, data['actors_optimizer']):
            optimizer.load_state_dict(state_dict)

        for optimizer, state_dict in zip(self.critics_optimizer, data['critics_optimizer']):
            optimizer.load_state_dict(state_dict)
    
    def log_network_stats(self, writer: SummaryWriter, time_step: int, log_interval: int = 100):
        """Log detailed network statistics to TensorBoard.
        
        Parameters:
        -----------
        writer : SummaryWriter
            TensorBoard writer instance
        time_step : int
            Current time step for logging
        log_interval : int, default=100
            How often to log detailed stats (every N steps)
        """
        if writer is None or time_step % log_interval != 0:
            return
            
        # Only log detailed stats periodically to avoid TensorBoard overhead
        with torch.no_grad():
            # Log actor weight stats
            for i, actor in enumerate(self.actors):
                # Gradient norm
                total_grad_norm = 0
                for p in actor.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar(f'Gradients/Actor_{i}_norm', total_grad_norm, time_step)
                
                # Weight norms per layer
                for name, param in actor.named_parameters():
                    if 'weight' in name:
                        writer.add_scalar(f'Weights/Actor_{i}/{name}_norm', param.norm().item(), time_step)
            
            # Log reward critic weight stats
            for i, critic in enumerate(self.critics):
                # Gradient norm
                total_grad_norm = 0
                for p in critic.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar(f'Gradients/RewardCritic_{i}_norm', total_grad_norm, time_step)
                
                # Weight norms per layer
                for name, param in critic.named_parameters():
                    if 'weight' in name:
                        writer.add_scalar(f'Weights/RewardCritic_{i}/{name}_norm', param.norm().item(), time_step)
            
            # Log constraint critic weight stats
            for i, constraint_critic in enumerate(self.constraint_critics):
                # Gradient norm
                total_grad_norm = 0
                for p in constraint_critic.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar(f'Gradients/ConstraintCritic_{i}_norm', total_grad_norm, time_step)
                
                # Weight norms per layer
                for name, param in constraint_critic.named_parameters():
                    if 'weight' in name:
                        writer.add_scalar(f'Weights/ConstraintCritic_{i}/{name}_norm', param.norm().item(), time_step)
            
            # Log ratio of critic vs constraint critic gradient norms
            for i in range(len(self.critics)):
                reward_norm = 0
                for p in self.critics[i].parameters():
                    if p.grad is not None:
                        reward_norm += p.grad.data.norm(2).item() ** 2
                reward_norm = reward_norm ** 0.5
                
                constraint_norm = 0
                for p in self.constraint_critics[i].parameters():
                    if p.grad is not None:
                        constraint_norm += p.grad.data.norm(2).item() ** 2
                constraint_norm = constraint_norm ** 0.5
                
                # Avoid division by zero
                if constraint_norm > 1e-10:
                    ratio = reward_norm / constraint_norm
                    writer.add_scalar(f'Gradients/Reward_to_Constraint_Ratio_{i}', ratio, time_step)

    def learn(self, episodes=100, deterministic=False, deterministic_finish=False, keep_env_history=True, writer=None):
        """Train the agent for a specified number of episodes.

        Parameters
        ----------
        episodes : int, default = 100
            Number of training episodes.
        deterministic : bool, default = False,
            Whether to sample deterministic actions from policy.
        deterministic_finish : bool, default = False
            Whether to finish the training with a deterministic episode.
        writer : torch.utils.tensorboard.SummaryWriter, optional
            TensorBoard summary writer for logging.
        
        Returns
        -------
        reward_all : list of float
            Returns for all timesteps.
        avg_returns: float
            Average returns of these episodes.
        """

        reward_all = []
        constraint_all = []
        observations_all = []
        avg_episode_train_time = []
        kpis_all = []
        q_vals = []
        lambdas = []
        all_rewards = []
        all_constraints = []

        for i in range(episodes):
            # Reset environment at the start of each episode
            self.env.reset()
            self.reset()

            cumulative_return = np.zeros(len(self.env.buildings))
            start = time.time()

            # Determine if this episode should be deterministic
            is_deterministic = deterministic
            if i == episodes - 1 and deterministic_finish:
                is_deterministic = True

            # Collect trajectories
            while not self.env.done:
                # Get current observations and actions
                observations = self.env.observations
                actions = self.predict(observations, deterministic=is_deterministic)
                
                # Get constraint values for metrics
                constraint_vals = None
                flat_constraints = []

                with torch.no_grad():
                    for j, b in enumerate(self.env.buildings):
                        if b.chargers:
                            for charger_index, charger in reversed(list(enumerate(b.chargers))):
                                # If no EV is connected, set action to 0
                                if not charger.connected_ev:
                                    actions[j][-charger_index - 1] = 0.0001

                # Step the environment
                next_observations, reward, done, _ = self.env.step(actions)
                
                # Calculate constraint values for logging
                with torch.no_grad():
                    encoded_obs = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
                    encoded_next_obs = [self.get_encoded_observations(i, next_obs) for i, next_obs in enumerate(next_observations)]
                    constraint_vals = self.get_constraint_cost(observations, actions, next_observations)
                    for c in constraint_vals:
                        flat_constraints.extend(c)
                    all_constraints.extend(flat_constraints)

                # Update the agent's parameters
                loss_vals = self.update(observations, actions, reward, next_observations, done)
                
                # Store reward values for monitoring
                all_rewards.extend(reward)

                # Call scale monitoring callback if it exists
                if self.scale_monitor_callback is not None and len(all_rewards) > 0 and len(all_constraints) > 0:
                    self.scale_monitor_callback(all_rewards, all_constraints, self.time_step)
                    
                cumulative_return[:] += np.array(reward)
                
                # Log training metrics to TensorBoard
                if writer and loss_vals is not None and loss_vals[0] is not None and loss_vals[1] is not None:
                    for i, (constraint_loss, critic_loss, lambda_val) in enumerate(zip(loss_vals[0], loss_vals[1], loss_vals[2])):
                        writer.add_scalar(f'Loss/ConstraintCritic/Agent_{i}', constraint_loss, self.time_step)
                        writer.add_scalar(f'Loss/Critic/Agent_{i}', critic_loss, self.time_step)
                        writer.add_scalar(f'Lambda/Agent_{i}', lambda_val, self.time_step)
                        if i == 0:
                            lambdas.append(lambda_val)
                    
                    writer.add_scalar('Reward/Step', sum(reward), self.time_step)
                    writer.add_scalar('Constraint/Step', sum(flat_constraints) if constraint_vals is not None else 0, self.time_step)
                    
                    # Log detailed network statistics periodically
                    self.log_network_stats(writer, self.time_step, log_interval=100)

            end = time.time()
            avg_episode_train_time.append(end - start)

            kpis = self.env.evaluate()
            
            observations_all.append(observations)
            kpis_all.append(kpis)
            constraint_all.append(all_constraints)
            reward_all.append(list(cumulative_return))
            
            print('episode: ', i)
            for b in range(len(self.env.buildings)):
                print(f"Building {b} - Reward: {cumulative_return[b]:.4f}")
                
            if writer:
                writer.add_scalar('Reward/Episode', np.sum(cumulative_return[:]), i)
                # Log per-building rewards
                for b in range(len(self.env.buildings)):
                    writer.add_scalar(f'Reward/Building_{b}', cumulative_return[b], i)
                
                # Log average constraint value per episode
                if len(all_constraints) > 0:
                    #writer.add_scalar('Constraint/Episode_Avg', np.mean(all_constraints), i)
                    #writer.add_scalar('Constraint/Episode_Avg', np.mean(all_constraints), i)
                    if torch.is_tensor(all_constraints[0]):
                        constraint_mean =torch.mean(torch.stack(all_constraints)).cpu().numpy()
                    else:
                        constraint_mean = np.mean(all_constraints)
                    writer.add_scalar('Constraint/Episode_Avg', constraint_mean, i)
                
                # Log Lagrangian multipliers
                for j, lambda_val in enumerate(self.lagrangians):
                    writer.add_scalar(f'Lambda_Episode/Agent_{j}', lambda_val.item(), i)

        return reward_all, avg_episode_train_time, kpis_all, observations_all, constraint_all, q_vals, lambdas

class MADDPGRBC(MADDPG):
    r"""Uses :py:class:`citylearn.agents.rbc.RBC` to select action during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    rbc: RBC
        :py:class:`citylearn.agents.rbc.RBC` or child class, used to select actions during exploration.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, rbc: RBC = None, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.rbc = RBC(env, **kwargs) if rbc is None else rbc

    def get_exploration_prediction(self, states: List[float]) -> List[float]:
        """Return actions using :class:`RBC`.

        Returns
        -------
        actions: List[float]
            Action values.
        """

        #print("V2G RBC") #commented NEW
        return self.rbc.predict(states)


class MADDPGHourRBC(MADDPGRBC):
    r"""Uses :py:class:`citylearn.agents.rbc.HourRBC` to select action during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.rbc = OptimizedRBC(env, **kwargs)


class MADDPGBasicRBC(MADDPGRBC):
    r"""Uses :py:class:`citylearn.agents.rbc.BasicRBC` to select action during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.rbc = BasicRBC(env, **kwargs)


class MADDPGOptimizedRBC(MADDPGRBC):
    r"""Uses :py:class:`citylearn.agents.rbc.OptimizedRBC` to select action during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.rbc = OptimizedRBC(env, **kwargs)


class MADDPGBasicBatteryRBC(MADDPGRBC):
    r"""Uses :py:class:`citylearn.agents.rbc.BasicBatteryRBC` to select action during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.rbc = BasicBatteryRBC(env, **kwargs)

class MADDPGV2GRBC(MADDPGRBC):
    r"""Uses :py:class:`citylearn.agents.rbc.V2GRBC` to select action during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.rbc = V2GRBC(env, **kwargs)

