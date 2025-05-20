import inspect
import logging
import os
from pathlib import Path
import pickle
from typing import Any, List, Mapping, Tuple, Union
from gym import spaces
from citylearn.base import Environment
from citylearn.citylearn import CityLearnEnv
import time
from torch.utils.tensorboard import SummaryWriter


LOGGER = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True

class Agent(Environment):
    r"""Base agent class.

    Parameters
    ----------
    env : CityLearnEnv
        CityLearn environment.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        self.env = env
        self.observation_names = self.env.observation_names
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.building_information = self.env.get_building_information()

        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key:value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))
            
        }
        super().__init__(**kwargs)

    @property
    def observation_names(self) -> List[List[str]]:
        """Names of active observations that can be used to map observation values."""

        return self.__observation_names

    @property
    def observation_space(self) -> List[spaces.Box]:
        """Format of valid observations."""

        return self.__observation_space

    @property
    def action_space(self) -> List[spaces.Box]:
        """Format of valid actions."""

        return self.__action_space

    @property
    def building_information(self) -> List[Mapping[str, Any]]:
        """Building metadata."""

        return self.__building_information

    @property
    def action_dimension(self) -> List[int]:
        """Number of returned actions."""

        return [s.shape[0] for s in self.action_space]

    @property
    def actions(self) -> List[List[List[Any]]]:
        """Action history/time series."""

        return self.__actions

    @observation_names.setter
    def observation_names(self, observation_names: List[List[str]]):
        self.__observation_names = observation_names

    @observation_space.setter
    def observation_space(self, observation_space: List[spaces.Box]):
        self.__observation_space = observation_space

    @action_space.setter
    def action_space(self, action_space: List[spaces.Box]):
        self.__action_space = action_space

    @building_information.setter
    def building_information(self, building_information: List[Mapping[str, Any]]):
        self.__building_information = building_information

    @actions.setter
    def actions(self, actions: List[List[Any]]):
        for i in range(len(self.action_space)):
            self.__actions[i][self.time_step] = actions[i]

    def learn(
            self, episodes: int = None, keep_env_history: bool = None, env_history_directory: Union[str, Path] = None, 
            deterministic: bool = None, deterministic_finish: bool = None, logging_level: int = None,
            writer: SummaryWriter = None
        ):
        """Train agent with optional TensorBoard logging."""

        episodes = 1 if episodes is None else episodes
        keep_env_history = False if keep_env_history is None else keep_env_history
        deterministic_finish = False if deterministic_finish is None else deterministic_finish
        deterministic = False if deterministic is None else deterministic
        self.__set_logger(logging_level)
    
        if keep_env_history:
            env_history_directory = Path(f'citylearn_learning_{self.env.uid}') if env_history_directory is None else env_history_directory
            os.makedirs(env_history_directory, exist_ok=True)
        else:
            pass
    
        constraint_all = []
        q_critic_all = []
        lambda_all = []
        rewards_all = []
        individual_runtimes_predict = []
        average_runtime = 0
        kpis_list = []
        observations_ep = []

        global_step = 0
    
        for episode in range(episodes):
            deterministic = deterministic or (deterministic_finish and episode >= episodes - 1)
            observations = self.env.reset()
            rewards_ep = []
            constraint_all_ep = []
            q_critic_all_ep = []
            lambda_all_ep = []
    
            while not self.env.done:
                print("\n \n ------TIME STEP------")
                print(f"{episode} - {self.env.time_step}")

                #print("------Predict------")
                observations_ep.append(observations)
                start_time = time.time() # Get the current time
                actions = self.predict(observations, deterministic=deterministic)
                end_time = time.time() # Get the current time again after the function has run
                individual_runtimes_predict.append(end_time - start_time)
    
                next_observations, rewards, _, _ = self.env.step(actions)
                rewards_ep.append(rewards)
                
                # update
                if not deterministic:
                    constraint_loss, critic_loss, lambda_loss = self.update(observations, actions, rewards, next_observations, done=self.env.done)
                    if critic_loss is not None:
                        constraint_all_ep.append(constraint_loss)
                        q_critic_all_ep.append(critic_loss)
                        lambda_all_ep.append(lambda_loss)
    
                        if writer is not None:
                            #step = self.env.time_step
                            #for i in range(self.num_agents):
                                #writer.add_scalar(f"Loss/Critic/Agent_{i}", critic_loss, step)
                                #writer.add_scalar(f"Loss/ConstraintCritic/Agent_{i}", constraint_loss, step)
                                #writer.add_scalar(f"Lagrangian/Agent_{i}", lambda_loss[i] if hasattr(lambda_loss, '__getitem__') else lambda_loss, step)
                            #avg_reward = sum(map(sum, rewards_ep)) / len(rewards_ep) if rewards_ep else 0
                            #writer.add_scalar("Reward/Step", avg_reward, step)
                            for i in range(self.num_agents):
                                writer.add_scalar(f"Loss/Critic/Agent_{i}", critic_loss, global_step)
                                writer.add_scalar(f"Loss/ConstraintCritic/Agent_{i}", constraint_loss, global_step)
                                writer.add_scalar(f"Lagrangian/Agent_{i}", lambda_loss[i] if hasattr(lambda_loss, '__getitem__') else lambda_loss, global_step)
                                # NEW: Log reward per agent
                                writer.add_scalar(f"Reward/Agent_{i}", rewards[i], global_step)
                            avg_reward = sum(map(sum, rewards_ep)) / len(rewards_ep) if rewards_ep else 0
                            writer.add_scalar("Reward/Step", avg_reward, global_step)
                        global_step += 1


    
                observations = [o for o in next_observations]
    
            average_runtime = sum(individual_runtimes_predict) / len(individual_runtimes_predict)
    
            kpis = self.env.evaluate().pivot(index='cost_function', columns='name', values='value')
            kpis = kpis.dropna(how='all')
            kpis_list.append(kpis)
    
            constraint_all.append(constraint_all_ep)
            q_critic_all.append(q_critic_all_ep)
            lambda_all.append(lambda_all_ep)
            rewards_ep = [reward for reward in rewards_ep if isinstance(reward, List)]
            rewards_all.append(rewards_ep)
    
            if keep_env_history:
                self.__save_env(episode, env_history_directory)
    
        return rewards_all, average_runtime, kpis_list, observations_ep, constraint_all, q_critic_all, lambda_all

    def get_env_history(self, directory: Union[str, Path], episodes: List[int] = None) -> Tuple[CityLearnEnv]:
        """Return tuple of :py:class:`citylearn.citylearn.CityLearnEnv` objects at terminal point for simulated episodes.

        Parameters
        ----------
        directory: Union[str, Path]
            Directory path where :py:class:`citylearn.citylearn.CityLearnEnv` pickled files are stored.
        episodes: List[int], optional
            Episodes whose environment should be returned. If None, all environments for all episodes
            are returned.

        Returns
        -------
        env_history: Tuple[CityLearnEnv]
            :py:class:`citylearn.citylearn.CityLearnEnv` objects.
        """
        
        env_history = ()
        episodes = sorted([
            int(f.split(directory)[-1].split('.')[0]) for f in os.listdir(directory) if f.endswith('.pkl')
        ]) if episodes is None else episodes

        for episode in episodes:
            filepath = os.path.join(directory, f'{int(episode)}.pkl')

            with (open(filepath, 'rb')) as f:
                env_history += (pickle.load(f),)

        return env_history

    def __save_env(self, episode: int, directory: Path):
        """Save current environment state to pickle file."""

        filepath = os.path.join(directory, f'{int(episode)}.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump(self.env, f)


    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for current time step.

        Return randomly sampled actions from `action_space`.
        
        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[List[float]]
            Action values
        """
        
        actions = [list(s.sample()) for s in self.action_space]
        self.actions = actions
        self.next_time_step()
        return actions
    
    def __set_logger(self, logging_level: int = None):
        """Set logging level."""

        logging_level = 30 if logging_level is None else logging_level
        assert logging_level >= 0, 'logging_level must be >= 0'
        LOGGER.setLevel(logging_level)

    def update(self, *args, **kwargs):
        """Update replay buffer and networks.
        
        Notes
        -----
        This implementation does nothing but is kept to keep the API for all agents similar during simulation.
        """

        pass

    def next_time_step(self):
        super().next_time_step()

        for i in range(len(self.action_space)):
            self.__actions[i].append([])

    def reset(self):
        super().reset()
        self.__actions = [[[]] for _ in self.action_space]