from typing import Any, List
from citylearn.preprocessing import Encoder, NoNormalization, PeriodicNormalization, OnehotEncoding, RemoveFeature, Normalize
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

class MADDPG(RLC):
    """Optimized version of MADDPG with improved GPU utilization and parallel processing."""
    
    def __init__(self, env: CityLearnEnv, actor_units: list = [256, 128], critic_units: list = [256, 128],
                 buffer_size: int = int(1e5), batch_size: int = 1024, gamma: float = 0.99, sigma=0.2,
                 target_update_interval: int = 2, lr_actor: float = 1e-5, lr_critic: float = 1e-4,
                 lr_dual: float = 1e-5, steps_between_training_updates: int = 5, decay_percentage=0.995, tau=1e-3,
                 target_network=False, *args, **kwargs):
        
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
        print(f"Using device: {self.device}")

        self.seed = random.randint(0, 100_000_000)
        self.actor_units = actor_units
        self.critic_units = critic_units
        self.tau = tau
        self.sigma = sigma

        # Initialize actors and critics
        self.actors = [
            Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, actor_units).to(
                self.device) for i in range(len(self.action_space))
        ]
        
        self.critics = [
            Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_units).to(
                self.device) for _ in range(len(self.action_space))
        ]

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

        # Initialize optimizers
        self.actors_optimizer = [torch.optim.SGD(self.actors[i].parameters(), lr=lr_actor) for i in
                               range(len(self.action_space))]
        self.critics_optimizer = [torch.optim.SGD(self.critics[i].parameters(), lr=lr_critic) for i in
                                range(len(self.action_space))]

        # Initialize constraint critics and their optimizers
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
                                  range(len(self.action_space))]

        # Initialize Lagrangian parameter and its optimizer
        self.lagrangian = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.lambda_optimizer = torch.optim.SGD([self.lagrangian], lr=lr_dual)

        # Initialize other parameters
        decay_factor = decay_percentage ** (1/self.env.time_steps)
        self.target_update_interval = target_update_interval
        self.steps_between_training_updates = steps_between_training_updates
        self.scaler = GradScaler()
        self.exploration_done = False

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

    def compute_constraint_cost(self, agent_num, actions):
        """Computes the constraint cost for a given agent based on EV charging power limits."""
        cost = []
        building = self.env.buildings[agent_num]
        for action in actions:
            sample_cost = 0.0
            if building.chargers:
                for j, charger in enumerate(building.chargers):
                    act_value = action[j].item()
                    real_power = act_value * charger.nominal_power
                    if real_power > charger.max_charging_power or real_power < charger.min_charging_power:
                        sample_cost += 1.0
            cost.append(sample_cost)
        return torch.tensor(cost, dtype=torch.float32, device=self.device).view(-1, 1)

    def get_constraint_cost(self, observations, actions, next_observations):
        costs = []
        for agent_num, (obs, action, next_obs) in enumerate(zip(observations, actions, next_observations)):
            sample_cost = 0.0
            building = self.env.buildings[agent_num]
            if building.chargers:
                for j, charger in enumerate(building.chargers):
                    act_value = action[j]
                    real_power = act_value * charger.nominal_power
                    if real_power > charger.max_charging_power or real_power < charger.min_charging_power:
                        sample_cost += 1
            costs.append([sample_cost])
        return torch.tensor(costs, dtype=torch.float32, device=self.device).view(-1, 1)

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

    def soft_update(self, local_model, target_model, tau=1e-3):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def predict(self, observations, deterministic=False):
        """Optimized predict method with efficient GPU usage."""
        # Pre-encode observations
        encoded_observations = torch.stack([
            torch.FloatTensor(self.get_encoded_observations(i, obs)).to(self.device)
            for i, obs in enumerate(observations)
        ])
        
        actions_return = None
        if self.time_step > self.end_exploration_time_step or deterministic:
            if deterministic:
                actions_return = self.predict_deterministic(encoded_observations)
            else:
                actions_return = self.get_exploration_prediction(observations)
        else:
            actions_return = self.get_exploration_prediction(observations)
        
        self.next_time_step()
        return actions_return

    def predict_deterministic(self, encoded_observations):
        """Optimized deterministic prediction with efficient GPU usage."""
        with torch.no_grad():
            actions = torch.stack([
                actor(obs) for actor, obs in zip(self.actors, encoded_observations)
            ])
            return actions.cpu().numpy()

    def get_exploration_prediction(self, states: List[List[float]]) -> List[float]:
        """Get exploration prediction with noise."""
        # Pre-encode states for deterministic actions
        encoded_states = [self.get_encoded_observations(i, state) for i, state in enumerate(states)]
        deterministic_actions = self.predict_deterministic(encoded_states)

        # Generate random noise
        random_noises = []
        for action in deterministic_actions:
            bias = 0.3
            noise = np.random.normal(scale=self.sigma) - bias
            random_noises.append(noise)

        actions = [noise + action for action, noise in zip(deterministic_actions, random_noises)]
        clipped_actions = [np.clip(action, -1, 1) for action in actions]
        actions_return = [action.tolist() for action in clipped_actions]

        # Apply hard constraints to exploration
        for i, b in enumerate(self.env.buildings):
            if b.chargers:
                for charger_index, charger in reversed(list(enumerate(b.chargers))):
                    if not charger.connected_ev:
                        actions_return[i][-charger_index - 1] = 0.0001

        return actions_return

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        """Get encoded observations."""
        return np.array([j for j in np.hstack(self.encoders[index]*np.array(observations, dtype=float)) if j != None], dtype=float)

    def reset(self):
        """Reset the agent."""
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