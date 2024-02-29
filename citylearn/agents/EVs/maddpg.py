from typing import Any, List
from citylearn.preprocessing import Encoder, NoNormalization, PeriodicNormalization, OnehotEncoding ,RemoveFeature, Normalize
import numpy as np
import torch
import torch.nn.functional as F
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from citylearn.rl import Actor, Critic, OUNoise, ReplayBuffer1
import random
import numpy.typing as npt
import timeit
from torch.cuda.amp import autocast, GradScaler
from citylearn.agents.rbc import RBC, BasicBatteryRBC, BasicRBC, V2GRBC, OptimizedRBC
from citylearn.agents.rlc import RLC
from torch.utils.tensorboard import SummaryWriter
import pickle

class MADDPG(RLC):
    def __init__(self, env: CityLearnEnv, actor_units: list = [256, 128], critic_units: list = [256, 128],
                 buffer_size: int = int(1e5), batch_size: int = 128, gamma: float = 0.99, sigma=0.2,
                 target_update_interval: int = 2, lr_actor: float = 1e-5, lr_critic: float = 1e-4,
                 steps_between_training_updates: int = 5, decay_percentage=0.995, tau=1e-3, *args, **kwargs):

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
        print(self.device)

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

        # Initialize target networks
        self.actors_target = [
            Actor(self.observation_dimension[i], self.action_space[i].shape[0], self.seed, actor_units).to(
                self.device) for i in range(len(self.action_space))
        ]
        self.critics_target = [
            Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_units).to(
                self.device) for _ in range(len(self.action_space))
        ]

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in
                                 range(len(self.action_space))]
        self.critics_optimizer = [torch.optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in
                                  range(len(self.action_space))]

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
        agent.actors_optimizer = [torch.optim.Adam(actor.parameters()) for actor in agent.actors]
        agent.critics_optimizer = [torch.optim.Adam(critic.parameters()) for critic in agent.critics]
        for optimizer, state_dict in zip(agent.actors_optimizer, data.get('actors_optimizer', [])):
            optimizer.load_state_dict(state_dict)
        for optimizer, state_dict in zip(agent.critics_optimizer, data.get('critics_optimizer', [])):
            optimizer.load_state_dict(state_dict)

        return agent


    def update(self, observations, actions, reward, next_observations, done):
        self.replay_buffer.push(observations, actions, reward, next_observations, done)

        if len(self.replay_buffer) < self.batch_size:
            print("returned due to buffer")
            return

        if not self.exploration_done:
            if self.time_step < self.end_exploration_time_step:
                print("returned due to minor")
                return
            elif self.time_step == self.end_exploration_time_step:
                self.exploration_done = True
                print("Ended exploration")
                return

        if self.time_step % self.steps_between_training_updates != 0:
            print("Not time to train")
            return

        print("training")
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.replay_buffer.sample(
            self.batch_size)

        obs_tensors = []
        next_obs_tensors = []
        actions_tensors = []
        reward_tensors = []
        dones_tensors = []

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
            reward_tensors.append(
                torch.tensor(rewards_batch[agent_num], dtype=torch.float32).to(self.device).view(-1, 1))
            dones_tensors.append(torch.tensor(dones_batch[agent_num], dtype=torch.float32).to(self.device).view(-1, 1))


        obs_full = torch.cat(obs_tensors, dim=1)
        next_obs_full = torch.cat(next_obs_tensors, dim=1)
        action_full = torch.cat(actions_tensors, dim=1)

        for agent_num, (actor, critic, actor_target, critic_target, actor_optim, critic_optim) in enumerate(
                zip(self.actors, self.critics, self.actors_target, self.critics_target, self.actors_optimizer,
                    self.critics_optimizer)):

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
                # Update actor
                predicted_actions = [self.actors[i](obs_tensors[i]) for i in range(self.num_agents)]
                predicted_actions_full = torch.cat(predicted_actions, dim=1)
                actor_loss = -critic(obs_full, predicted_actions_full).mean()

            self.scaler.scale(actor_loss).backward()
            self.scaler.step(actor_optim)
            actor_optim.zero_grad()
            self.scaler.update()

            if self.time_step % self.target_update_interval == 0:
                # Update target networks
                self.soft_update(critic, critic_target, self.tau)
                self.soft_update(actor, actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_deterministic_actions(self, observations):
        with torch.no_grad():
            encoded_observations = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
            to_return = [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                    for actor, obs in zip(self.actors, encoded_observations)]
            return to_return

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

    def get_exploration_prediction(self, states: List[List[float]]) -> List[float]:
        """Return random actions`.

        Returns
        -------
        actions: List[float]
            Action values.
        """
        #actions = [self.noise[i].sample() + action for i, action in
        #           enumerate(self.get_deterministic_actions(states))]

        deterministic_actions = self.get_deterministic_actions(states)

        # Generate random noise and print its sign for each action
        random_noises = []
        for action in deterministic_actions:
            bias = 0.3
            noise = np.random.normal(scale=self.sigma) - bias
            random_noises.append(noise)

        actions = [noise + action for action, noise in zip(deterministic_actions, random_noises)]
        clipped_actions = [np.clip(action, -1, 1) for action in actions]
        actions_return = [action.tolist() for action in clipped_actions]

        #Hard Constraints to exploration
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

        print("V2G RBC")
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