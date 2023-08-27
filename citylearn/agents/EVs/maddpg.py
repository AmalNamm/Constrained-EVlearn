from typing import Any, List

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
from citylearn.agents.rbc import RBC, BasicBatteryRBC, BasicRBC, HourRBC, OptimizedRBC
from citylearn.agents.rlc import RLC
from torch.utils.tensorboard import SummaryWriter

class MADDPG(RLC):
    def __init__(self, env: CityLearnEnv, actor_units: tuple = (256, 128), critic_units: tuple = (256, 128),
                 buffer_size: int = int(1e5), batch_size: int = 100, gamma: float = 0.99,
                 target_update_interval: int = 1000, lr_actor: float = 1e-4, lr_critic: float = 1e-3, update_every: int = 5,
                 *args, **kwargs):
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

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in
                                 range(len(self.action_space))]
        self.critics_optimizer = [torch.optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in
                                  range(len(self.action_space))]

        self.noise = [OUNoise(self.action_space[i].shape[0], self.seed) for i in range(len(self.action_space))]

        self.target_update_interval = target_update_interval
        self.update_every = update_every
        self.timestep = 0  # Keep track of timesteps
        self.scaler = GradScaler()

    def update(self, observations, actions, reward, next_observations, done):
        self.replay_buffer.push(observations, actions, reward, next_observations, done)

        print("Reward MADDPG")
        print(reward)

        if len(self.replay_buffer) < self.batch_size or self.timestep < self.end_exploration_time_step:
            print("returned")
            return

        if self.timestep % self.update_every != 0:
            print("returned 2")
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

            if self.timestep % self.target_update_interval == 0:
                # Update target networks
                self.soft_update(critic, critic_target)
                self.soft_update(actor, actor_target)

    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def predict(self, observations, deterministic=False):

        if self.timestep > self.end_exploration_time_step or deterministic:
            print("ALI")
            with torch.no_grad():
                encoded_observations = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
                if deterministic:
                    self.timestep += 1
                    return [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                            for actor, obs in zip(self.actors, encoded_observations)]
                else:
                    return [self.noise[i].sample() + action
                            for i, action in enumerate(self.predict(observations, True))]
        else:
            print("AQUI")
            self.timestep += 1
            return self.get_exploration_prediction(observations)


    def get_exploration_prediction(self, states: List[float]) -> List[float]:
        """Return random actions`.

        Returns
        -------
        actions: List[float]
            Action values.
        """

        return [list(self.action_scaling_coefficient*s.sample()) for s in self.action_space]

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(self.encoders[index]*np.array(observations, dtype=float)) if j != None], dtype = float)


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

        print("Optmized RBC")
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