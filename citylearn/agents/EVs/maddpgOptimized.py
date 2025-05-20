from typing import Any, List
from citylearn.preprocessing import Encoder, NoNormalization, PeriodicNormalization, OnehotEncoding ,RemoveFeature, Normalize
import numpy as np
import torch
import torch.nn.functional as F
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
#from citylearn.rl import Actor, Critic, OUNoise, ReplayBuffer2
from citylearn.rl_tgelu import Actor, Critic, ReplayBuffer2
import random
import numpy.typing as npt
import timeit
from torch.cuda.amp import autocast, GradScaler
from citylearn.agents.rbc import RBC, BasicBatteryRBC, BasicRBC, V2GRBC, OptimizedRBC
from torch.utils.tensorboard.writer import SummaryWriter
import pickle

import csv
import os

class MADDPG(RLC):
    def __init__(self, env: CityLearnEnv, actor_units: list = [256, 128], critic_units: list = [256, 128],
                 buffer_size: int = int(1e5), batch_size: int = 1024, gamma: float = 0.99, sigma=0.2,
                 target_update_interval: int = 2, lr_actor: float = 1e-5, lr_critic: float = 1e-4,
                 lr_dual: float = 1e-5, steps_between_training_updates: int = 5, decay_percentage=0.995, tau=1e-3,
                 target_network=False, *args, **kwargs): ###NEW ### Added lr_dual: float = 1e-5

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
        self.tau = tau
        self.sigma = sigma

        ##added 06.05.2025 
        #set up a CSV log file
        self.lagrangian_logfile = "lagrangian_monitoring_New_Cost_inequality_Layer_Update.csv"
        if not os.path.exists(self.lagrangian_logfile):
            with open(self.lagrangian_logfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "agent", "lagrangian", "mean_constraint_violation"])


        # Initialize actors and critics
        # Each actor network handles its agent's local observation and action space, learning its own policy.
        self.actors = [
            Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, actor_units).to(
                self.device) for i in range(len(self.action_space))
        ]  ##decentrailised Actor 20.02.2025
        
        #Each critic network, receiving global information, can assess the overall quality of the joint actions taken by all agents, which is particularly useful in cooperative or competitive multi-agent settings.
        
        self.critics = [
            Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_units).to(
                self.device) for _ in range(len(self.action_space))
        ] ##centrailised Critic 20.02.2025

        # Initialize target networks if target_network is True
        if self.target_network:
            self.actors_target = [
                Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, actor_units).to(
                    self.device) for i in range(len(self.action_space))
            ]
            self.critics_target = [
                Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_units).to(
                    self.device) for _ in range(len(self.action_space))
            ]

        self.actors_optimizer = [torch.optim.SGD(self.actors[i].parameters(), lr=lr_actor) for i in
                                 range(len(self.action_space))]
        self.critics_optimizer = [torch.optim.SGD(self.critics[i].parameters(), lr=lr_critic) for i in
                                  range(len(self.action_space))]

        ### NEW ### Constraint Critic and its target networks ***
        #***
        self.constraint_critics = [
            Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_units).to(
                self.device) for _ in range(len(self.action_space))
        ]

        if self.target_network:
            self.constraint_critics_target = [
                Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_units).to(
                    self.device) for _ in range(len(self.action_space))
            ]

        self.constraint_critics_optimizer = [torch.optim.SGD(self.constraint_critics[i].parameters(), lr=lr_critic) for i in
                                  range(len(self.action_space))] ###NEW TO DO : Should we use the same lr_critic ; learning rate for both critics? 

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
    
        cons = self.get_constraint_cost(observations, actions, next_observations)
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

        obs_batch, actions_batch, rewards_batch, next_obs_batch, constraint_batch, dones_batch = self.replay_buffer.sample(
            self.batch_size)

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

        obs_full = torch.cat(obs_tensors, dim=1)
        next_obs_full = torch.cat(next_obs_tensors, dim=1)
        action_full = torch.cat(actions_tensors, dim=1)

        global_constraint_costs = []

        # If not using target networks, use main networks as targets
        if not self.target_network:
            self.actors_target = self.actors
            self.critics_target = self.critics
            self.constraint_critics_target = self.constraint_critics

        for agent_num, (actor, critic, constraint_critic, actor_target, critic_target, constraint_critic_target, 
                        actor_optim, critic_optim, constraint_optim) in enumerate(
                zip(self.actors, self.critics, self.constraint_critics, 
                    self.actors_target, self.critics_target, self.constraint_critics_target,
                    self.actors_optimizer, self.critics_optimizer, self.constraint_critics_optimizer)):
            
            with autocast():
                # Update critic
                Q_expected = critic(obs_full, action_full)
                next_actions = [self.actors_target[i](next_obs_tensors[i]) for i in range(self.num_agents)]
                next_actions_full = torch.cat(next_actions, dim=1)
                Q_targets_next = critic_target(next_obs_full, next_actions_full)
                Q_targets = reward_tensors[agent_num] + (self.gamma * Q_targets_next * (1 - dones_tensors[agent_num]))
                critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

            self.scaler.scale(critic_loss).backward()
            self.scaler.step(critic_optim)
            critic_optim.zero_grad()
            self.scaler.update()

            with autocast():
                # Update constraint critic
                constraint_expected = constraint_critic(obs_full, action_full)
                constraint_cost = constraint_tensor[agent_num]
                next_actions = [self.actors_target[i](next_obs_tensors[i]) for i in range(self.num_agents)]
                next_actions_full = torch.cat(next_actions, dim=1)
                constraint_targets_next = constraint_critic_target(next_obs_full, next_actions_full)
                constraint_targets = constraint_cost + (self.gamma * constraint_targets_next * (1 - dones_tensors[agent_num]))
                constraint_loss = F.mse_loss(constraint_expected, constraint_targets.detach())

            self.scaler.scale(constraint_loss).backward()
            self.scaler.step(constraint_optim)
            constraint_optim.zero_grad()
            self.scaler.update()

            global_constraint_costs.append(constraint_cost)

            with autocast():
                # Update actor
                predicted_actions = [self.actors[i](obs_tensors[i]) for i in range(self.num_agents)]
                predicted_actions_full = torch.cat(predicted_actions, dim=1)
                #actor_loss = -critic(obs_full, predicted_actions_full).mean() + self.lagrangian * constraint_critic(obs_full, predicted_actions_full).mean()
                
                ###Use Per-Agent Multipliers : updated 05.05.2025
                #actor_loss = -critic(obs_full, predicted_actions_full).mean() + \
                #self.lagrangians[agent_num] * 100* constraint_critic(obs_full, predicted_actions_full).mean() #added 100 to experiment and scale

                actor_loss = -critic(obs_full, predicted_actions_full).mean() + \
                self.lagrangians[agent_num] * constraint_critic(obs_full, predicted_actions_full).mean() #added 100 to experiment and scale



            self.scaler.scale(actor_loss).backward()
            self.scaler.step(actor_optim)
            actor_optim.zero_grad()
            self.scaler.update()

            # Update target networks if using them
            if self.target_network and self.time_step % self.target_update_interval == 0:
                self.soft_update(critic, critic_target, self.tau)
                self.soft_update(actor, actor_target, self.tau)
                self.soft_update(constraint_critic, constraint_critic_target, self.tau)

        # Update Lagrangian multiplier
        #global_constraint_cost = torch.mean(torch.cat(global_constraint_costs, dim=0))
        
        
        #mean_constraint_violations = self.constraint_buffer.mean(dim=0)  # shape: [num_agents]
        mean_constraint_violations = torch.stack([cons_tensor.mean() for cons_tensor in constraint_tensor])


        #with autocast():
         #   lambda_loss = -self.lagrangian * global_constraint_cost
        #self.scaler.scale(lambda_loss).backward()
        #self.scaler.step(self.lambda_optimizer)
        #self.lambda_optimizer.zero_grad()
        #self.scaler.update()
        
        #update each lagrangian one individually: updated 05.05.2025
        with torch.no_grad():
            self.lagrangians += self.lr_dual * mean_constraint_violations  # size: [num_agents]
            self.lagrangians.clamp_(min=0.0)

        # In your update function, after updating self.lagrangians:
        for i in range(self.num_agents):
            with open(self.lagrangian_logfile, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.time_step,  # or global_step if you have one
                    i,
                    float(self.lagrangians[i].item()),
                    float(mean_constraint_violations[i].item())
                ])


        #return constraint_loss.item(), critic_loss.item(), lambda_loss.item()
        return constraint_loss.item(), critic_loss.item(), self.lagrangians.clone().detach().cpu().numpy()


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

# Example of optimized GPU usage
class OptimizedMADDPG(MADDPGOptimizedRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-allocate GPU tensors
        self.obs_buffer = torch.zeros((self.batch_size, sum(self.observation_dimension)), 
                                    device=self.device)
        self.action_buffer = torch.zeros((self.batch_size, sum(self.action_dimension)), 
                                       device=self.device)
        self.reward_buffer = torch.zeros((self.batch_size, self.num_agents), 
                                       device=self.device)
        
    def update(self, observations, actions, reward, next_observations, done):
        # Batch encode on GPU
        encoded_obs = torch.stack([
            torch.FloatTensor(self.get_encoded_observations(i, obs)).to(self.device)
            for i, obs in enumerate(observations)
        ])
        
        # Keep data on GPU
        self.obs_buffer.copy_(encoded_obs)
        self.action_buffer.copy_(torch.FloatTensor(actions).to(self.device))
        self.reward_buffer.copy_(torch.FloatTensor(reward).to(self.device))
        
        # Process all agents in parallel
        with autocast():
            # All networks process data simultaneously
            Q_values = torch.stack([critic(self.obs_buffer, self.action_buffer) 
                                  for critic in self.critics])
            constraint_values = torch.stack([constraint_critic(self.obs_buffer, self.action_buffer) 
                                          for constraint_critic in self.constraint_critics])
            
        # Update all networks in parallel
        self.optimize_networks(Q_values, constraint_values)


        #Why this helps:
        #Instead of creating new tensors for each batch, we reuse pre-allocated memory

        #Tensors are already on GPU, eliminating transfer overhead
        #Fixed memory layout improves cache utilization
        #Reduces memory fragmentation


class OptimizedMADDPG_V2(MADDPGOptimizedRBC):
    """Optimized version of MADDPG with improved GPU utilization and parallel processing."""
    
    def __init__(self, env: CityLearnEnv, actor_units: list = [256, 128], critic_units: list = [256, 128],
                 buffer_size: int = int(1e5), batch_size: int = 1024, gamma: float = 0.99, sigma=0.2,
                 target_update_interval: int = 2, lr_actor: float = 1e-5, lr_critic: float = 1e-4,
                 lr_dual: float = 1e-5, steps_between_training_updates: int = 5, decay_percentage=0.995, tau=1e-3,
                 target_network=False, *args, **kwargs):
        
        super().__init__(env, actor_units, critic_units, buffer_size, batch_size, gamma, sigma,
                        target_update_interval, lr_actor, lr_critic, lr_dual, steps_between_training_updates,
                        decay_percentage, tau, target_network, *args, **kwargs)
        
        # Pre-allocate GPU tensors for efficient memory usage
        self.obs_buffer = torch.zeros((self.batch_size, sum(self.observation_dimension)), 
                                    device=self.device)
        self.action_buffer = torch.zeros((self.batch_size, sum(self.action_dimension)), 
                                       device=self.device)
        self.reward_buffer = torch.zeros((self.batch_size, self.num_agents), 
                                       device=self.device)
        self.dones_buffer = torch.zeros((self.batch_size, self.num_agents), 
                                      device=self.device)
        self.constraint_buffer = torch.zeros((self.batch_size, self.num_agents), 
                                           device=self.device)
        
        # Initialize incremental mean calculator for constraint costs
        self.constraint_mean = self.IncrementalMean()
    
    class IncrementalMean:
        """Helper class for computing mean incrementally."""
        def __init__(self):
            self.mean = 0.0
            self.count = 0
            
        def update(self, new_value):
            self.count += 1
            self.mean = self.mean + (new_value - self.mean) / self.count
            return self.mean
    
    def update(self, observations, actions, reward, next_observations, done):
        """Optimized update method with improved GPU utilization."""
        # Pre-encode observations and move to GPU
        encoded_obs = torch.stack([
            torch.FloatTensor(self.get_encoded_observations(i, obs)).to(self.device)
            for i, obs in enumerate(observations)
        ])
        encoded_next_obs = torch.stack([
            torch.FloatTensor(self.get_encoded_observations(i, next_obs)).to(self.device)
            for i, next_obs in enumerate(next_observations)
        ])
        
        # Compute constraint costs
        cons = self.get_constraint_cost(observations, actions, next_observations)
        
        # Push to replay buffer
        self.replay_buffer.push(encoded_obs.cpu().numpy(), actions, reward, 
                              encoded_next_obs.cpu().numpy(), cons, done)
        
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
        
        # Sample from buffer and move to GPU efficiently
        obs_batch, actions_batch, rewards_batch, next_obs_batch, constraint_batch, dones_batch = \
            self.replay_buffer.sample(self.batch_size)
        
        # Efficient tensor creation and transfer
        self.obs_buffer.copy_(torch.FloatTensor(np.concatenate(obs_batch, axis=1)).to(self.device))
        self.action_buffer.copy_(torch.FloatTensor(np.concatenate(actions_batch, axis=1)).to(self.device))
        self.reward_buffer.copy_(torch.FloatTensor(np.stack(rewards_batch, axis=1)).to(self.device))
        self.dones_buffer.copy_(torch.FloatTensor(np.stack(dones_batch, axis=1)).to(self.device))
        self.constraint_buffer.copy_(torch.FloatTensor(np.stack(constraint_batch, axis=1)).to(self.device))
        
        # Process all networks in parallel
        with autocast():
            # Forward pass for all networks
            Q_values = torch.stack([critic(self.obs_buffer, self.action_buffer) 
                                  for critic in self.critics])
            constraint_values = torch.stack([constraint_critic(self.obs_buffer, self.action_buffer) 
                                          for constraint_critic in self.constraint_critics])
        
        # Optimize networks
        return self.optimize_networks(Q_values, constraint_values)
    
    def optimize_networks(self, Q_values, constraint_values):
        """Optimized parallel network updates for all agents."""
        # 1. Prepare all gradients at once
        with autocast():
            # Compute all losses in parallel
            critic_losses = []
            constraint_losses = []
            actor_losses = []
            
            for agent_num in range(self.num_agents):
                # Critic loss
                Q_expected = self.critics[agent_num](self.obs_buffer, self.action_buffer)
                Q_targets = self.reward_buffer[:, agent_num].unsqueeze(1) + \
                           (self.gamma * Q_values[agent_num] * (1 - self.dones_buffer[:, agent_num].unsqueeze(1)))
                critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
                critic_losses.append(critic_loss)
                
                # Constraint critic loss
                constraint_expected = self.constraint_critics[agent_num](self.obs_buffer, self.action_buffer)
                constraint_targets = self.constraint_buffer[:, agent_num].unsqueeze(1) + \
                                   (self.gamma * constraint_values[agent_num] * (1 - self.dones_buffer[:, agent_num].unsqueeze(1)))
                constraint_loss = F.mse_loss(constraint_expected, constraint_targets.detach())
                constraint_losses.append(constraint_loss)
                
                # Actor loss
                predicted_actions = self.actors[agent_num](self.obs_buffer)
                actor_loss = -self.critics[agent_num](self.obs_buffer, predicted_actions).mean() + \
                            self.lagrangian * self.constraint_critics[agent_num](self.obs_buffer, predicted_actions).mean()
                
                
                actor_losses.append(actor_loss)
        
        # 2. Update all networks in parallel
        # Critic updates
        for critic_loss, critic_optim in zip(critic_losses, self.critics_optimizer):
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(critic_optim)
            critic_optim.zero_grad()
        
        # Constraint critic updates
        for constraint_loss, constraint_optim in zip(constraint_losses, self.constraint_critics_optimizer):
            self.scaler.scale(constraint_loss).backward()
            self.scaler.step(constraint_optim)
            constraint_optim.zero_grad()
        
        # Actor updates
        for actor_loss, actor_optim in zip(actor_losses, self.actors_optimizer):
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(actor_optim)
            actor_optim.zero_grad()
        
        # 3. Update target networks if using them
        if self.target_network and self.time_step % self.target_update_interval == 0:
            for critic, critic_target, actor, actor_target, constraint_critic, constraint_critic_target in \
                zip(self.critics, self.critics_target, self.actors, self.actors_target, 
                    self.constraint_critics, self.constraint_critics_target):
                self.soft_update(critic, critic_target, self.tau)
                self.soft_update(actor, actor_target, self.tau)
                self.soft_update(constraint_critic, constraint_critic_target, self.tau)
        
        # 4. Update Lagrangian multiplier using incremental mean
        global_constraint_cost = self.constraint_mean.update(torch.mean(torch.stack(constraint_losses)))
        
        with autocast():
            lambda_loss = -self.lagrangian * global_constraint_cost
        self.scaler.scale(lambda_loss).backward()
        self.scaler.step(self.lambda_optimizer)
        self.lambda_optimizer.zero_grad()
        
        self.scaler.update()

        
        
        
        return (torch.mean(torch.stack(constraint_losses)).item(),
                torch.mean(torch.stack(critic_losses)).item(),
                lambda_loss.item())
    


    