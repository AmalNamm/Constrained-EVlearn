from typing import List, Tuple
import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.energy_model import ZERO_DIVISION_CAPACITY

class RewardFunction:
    r"""Base and default reward function class.

    The default reward is the electricity consumption from the grid at the current time step returned as a negative value.

    Parameters
    ----------
    env: citylearn.citylearn.CityLearnEnv
        CityLearn environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.

    Notes
    -----
    Reward value is calculated as :math:`[\textrm{min}(-e_0, 0), \dots, \textrm{min}(-e_n, 0)]` 
    where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents.
    """
    
    def __init__(self, env: CityLearnEnv, **kwargs):
        self.env = env
        self.kwargs = kwargs

    @property
    def env(self) -> CityLearnEnv:
        """Simulation environment."""

        return self.__env

    @env.setter
    def env(self, env: CityLearnEnv):
        self.__env = env

    def calculate(self) -> List[float]:
        r"""Calculates reward.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        if self.env.central_agent:
            reward = [min(self.env.net_electricity_consumption[self.env.time_step]*-1, 0)]
        else:
            reward = [min(b.net_electricity_consumption[b.time_step]*-1, 0) for b in self.env.buildings]

        return reward

class MARL(RewardFunction):
    """MARL reward function class.

    Parameters
    ----------
    env: citylearn.citylearn.CityLearnEnv
        CityLearn environment.
    
    Notes
    -----
    Reward value is calculated as :math:`\textrm{sign}(-e) \times 0.01(e^2) \times \textrm{max}(0, E)`
    where :math:`e` is the building `electricity_consumption` and :math:`E` is the district `electricity_consumption`.
    """

    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        district_electricity_consumption = self.env.net_electricity_consumption[self.env.time_step]
        building_electricity_consumption = np.array([b.net_electricity_consumption[b.time_step]*-1 for b in self.env.buildings])
        reward_list = np.sign(building_electricity_consumption)*0.01*building_electricity_consumption**2*np.nanmax([0, district_electricity_consumption])

        if self.env.central_agent:
            reward = [reward_list.sum()]
        else:
            reward = reward_list.tolist()
        
        return reward

class IndependentSACReward(RewardFunction):
    """Recommended for use with the `SAC` controllers.
    
    Returned reward assumes that the building-agents act independently of each other, without sharing information through the reward.

    Parameters
    ----------
    env: citylearn.citylearn.CityLearnEnv
        CityLearn environment.

    Notes
    -----
    Reward value is calculated as :math:`[\textrm{min}(-e_0^3, 0), \dots, \textrm{min}(-e_n^3, 0)]` 
    where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents.
    """
    
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        reward_list = [min(b.net_electricity_consumption[b.time_step]*-1**3, 0) for b in self.env.buildings]

        if self.env.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward
    
class SolarPenaltyReward(RewardFunction):
    """The reward is designed to minimize electricity consumption and maximize solar generation to charge energy storage systems.

    The reward is calculated for each building, i and summed to provide the agent with a reward that is representative of all the
    building or buildings (in centralized case)it controls. It encourages net-zero energy use by penalizing grid load satisfaction 
    when there is energy in the enerygy storage systems as well as penalizing net export when the energy storage systems are not
    fully charged through the penalty term. There is neither penalty nor reward when the energy storage systems are fully charged
    during net export to the grid. Whereas, when the energy storage systems are charged to capacity and there is net import from the 
    grid the penalty is maximized.

    Parameters
    ----------
    env: citylearn.citylearn.CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        reward_list = []

        for b in self.env.buildings:
            e = b.net_electricity_consumption[-1]
            cc = b.cooling_storage.capacity
            hc = b.heating_storage.capacity
            dc = b.dhw_storage.capacity
            ec = b.electrical_storage.capacity_history[0]
            cs = b.cooling_storage.soc[-1]/cc
            hs = b.heating_storage.soc[-1]/hc
            ds = b.dhw_storage.soc[-1]/dc
            es = b.electrical_storage.soc[-1]/ec
            reward = 0.0
            reward += -(1.0 + np.sign(e)*cs)*abs(e) if cc > ZERO_DIVISION_CAPACITY else 0.0
            reward += -(1.0 + np.sign(e)*hs)*abs(e) if hc > ZERO_DIVISION_CAPACITY else 0.0
            reward += -(1.0 + np.sign(e)*ds)*abs(e) if dc > ZERO_DIVISION_CAPACITY else 0.0
            reward += -(1.0 + np.sign(e)*es)*abs(e) if ec > ZERO_DIVISION_CAPACITY else 0.0
            reward_list.append(reward)


        if self.env.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list
        
        return reward


class V2GPenaltyReward(RewardFunction):
    """Rewards with considerations for car charging and for MADDPG Mixed environments.

    Parameters
    ----------
    env: citylearn.citylearn.CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        self.rolling_window = []

    # Penalties
    PEAK_PERCENTAGE_THRESHOLD = 0.10  # Example: 10% increase over the average of last x timesteps.
    RAMPING_PERCENTAGE_THRESHOLD = 0.20  # Example: 20% increase in the change of consumption from average.
    PEAK_PENALTY_WEIGHT = 2
    RAMPING_PENALTY_WEIGHT = 1.5
    ENERGY_TRANSFER_BONUS = 0.5
    WINDOW_SIZE = 5  # Example: Last 5 timesteps

    PENALTY_NO_CAR_CHARGING = -100
    PENALTY_EXCEED_MAX_RATE = -50
    PENALTY_BATTERY_LIMITS = -30
    PENALTY_SOC_UNDER_5_10 = -10
    PENALTY_SOC_OVER_5_10 = -5

    REWARD_CLOSE_SOC = 10  # Reward value when SOC is close to what's requested

    def calculate_building_reward(self, b) -> float:
        """Calculate individual building reward."""
        net_energy = b.net_electricity_consumption[b.time_step]

        # Reward initialization
        reward = 0

        # Building reward calculation
        if b.reward_type == "C":  # Pricing-based reward
            if net_energy > 0:  # Consuming from the grid
                reward = -b.pricing[b.time_step] * net_energy
            else:  # Exporting to the grid
                reward = 0.80 * b.pricing[b.time_step] * abs(net_energy)
        elif b.reward_type == "G":  # Reducing carbon emissions
            reward = b.carbon_intensity[b.time_step] * (net_energy * -1)
        elif b.reward_type == "Z":  # Increasing zero net energy
            if net_energy > 0:  # The building is consuming more than it's producing
                reward = -net_energy
            else:  # The building is producing excess energy or is balanced
                reward = abs(net_energy) * 0.5  # Lesser reward for exporting TODO
        else:
            reward = net_energy * -1

        # Deducting EV penalties from the building reward
        reward += self.calculate_ev_penalty(b)

        return reward

    def calculate_community_reward(self, buildings, rewards) -> List[float]:
        """Calculate community building reward."""

        # Calculate the net energy of the entire community by summing the energy consumed/generated by each building.
        community_net_energy = sum(b.net_electricity_consumption[b.time_step] for b in buildings)

        # Update the rolling window of past net energies. This window keeps track of the last WINDOW_SIZE values.
        # If the window is already full (reached its WINDOW_SIZE), remove the oldest value.
        if len(self.rolling_window) >= self.WINDOW_SIZE:
            self.rolling_window.pop(0)
        # Append the current net energy to the rolling window.
        self.rolling_window.append(community_net_energy)

        # Calculate the average net energy consumption of the community over the past WINDOW_SIZE time steps.
        average_past_consumption = sum(self.rolling_window) / len(self.rolling_window)

        # Determine a dynamic peak threshold based on the average consumption plus a certain percentage.
        dynamic_peak_threshold = average_past_consumption * (1 + self.PEAK_PERCENTAGE_THRESHOLD)

        # Calculate the previous ramping (change in net energy from the last time step to the current time step).
        # If there's not enough data in the window, consider it as zero.
        if len(self.rolling_window) > 1:
            previous_ramping = community_net_energy - self.rolling_window[-2]
        else:
            previous_ramping = 0

        # Determine a dynamic ramping threshold based on the previous ramping value plus a certain percentage.
        dynamic_ramping_threshold = previous_ramping * (1 + self.RAMPING_PERCENTAGE_THRESHOLD)

        # Calculate the current ramping (change in net energy from the average of the window to the current time step).
        ramping = community_net_energy - average_past_consumption

        # Initialize the community reward to zero.
        community_reward = 0
        # Penalize if the community's net energy exceeds the dynamic peak threshold.
        if community_net_energy > dynamic_peak_threshold:
            community_reward -= (community_net_energy - dynamic_peak_threshold) * self.PEAK_PENALTY_WEIGHT

        # Penalize if the community's energy change rate (ramping) exceeds the dynamic ramping threshold.
        if abs(ramping) > dynamic_ramping_threshold:
            community_reward -= abs(ramping) * self.RAMPING_PENALTY_WEIGHT

        # Reward individual buildings that are exporting energy to the grid (assuming other buildings can use this exported energy).
        for b in buildings:
            if b.net_electricity_consumption[
                b.time_step] < 0:  # If a building is exporting energy (negative consumption).
                community_reward += abs(b.net_electricity_consumption[b.time_step]) * self.ENERGY_TRANSFER_BONUS

        # Combine the calculated community reward with the individual rewards of each building.
        updated_rewards = [r + community_reward for r in rewards]

        return updated_rewards

    def calculate_ev_penalty(self, b) -> float:
        """Calculate penalties based on EV specific logic."""
        penalty = 0
        if b.chargers:
            for c in b.chargers:
                last_connected_car = c.past_connected_evs[-2]
                last_charged_value = c.past_charging_action_values[-2]
                last_charged_value = last_charged_value * last_connected_car.battery.capacity

                # 1. Penalty for charging when no car is present
                if last_connected_car is None and last_charged_value != 0:
                    penalty += self.PENALTY_NO_CAR_CHARGING

                # 2. Penalty if the value of the charge action exceeds the charger's max charging rate
                if abs(last_charged_value) > c.max_charging_power or abs(last_charged_value) < c.max_discharging_power:
                    penalty += self.PENALTY_EXCEED_MAX_RATE

                # 3. Penalty for exceeding the battery's limits
                if last_connected_car is not None:
                    if last_connected_car.battery.soc[-2] + last_charged_value > last_connected_car.battery.capacity:
                        penalty += self.PENALTY_BATTERY_LIMITS
                    if last_connected_car.battery.soc[-2] + last_charged_value < last_connected_car.min_battery_soc:
                        penalty += self.PENALTY_BATTERY_LIMITS

                # 4. Penalties for SoC differences
                if last_connected_car is not None:
                    required_soc = last_connected_car.ev_simulation.required_soc_departure[-2]
                    actual_soc = last_connected_car.battery.soc[-2]
                    soc_diff = (actual_soc*100)/last_connected_car.battery.capacity - required_soc

                    # Penalize for lower SoC than required
                    if -10 < soc_diff <= -5:
                        penalty += self.PENALTY_SOC_UNDER_5_10
                    elif -25 < soc_diff <= -10:
                        penalty += 2 * self.PENALTY_SOC_UNDER_5_10
                    elif soc_diff <= -25:
                        penalty += 4 * self.PENALTY_SOC_UNDER_5_10

                    # Penalize for higher SoC than required
                    if 5 < soc_diff <= 10:
                        penalty += self.PENALTY_SOC_OVER_5_10
                    elif soc_diff > 10:
                        penalty += 2 * self.PENALTY_SOC_OVER_5_10

                    # Reward for leaving with SOC close to the requested value
                    if -5 < soc_diff <= 5:
                        penalty += self.REWARD_CLOSE_SOC

        return penalty

    def calculate(self) -> List[float]:
        reward_list = []

        for b in self.env.buildings:
            # Building reward calculation
            reward = self.calculate_building_reward(b)

            # Community Level Reward
            reward += self.WEIGHT_REWARD_COMMUNITY

            # EV specific penalties
            reward += self.calculate_ev_penalty(b)

            reward_list.append(reward)

        # Central agent reward aggregation
        if self.env.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward

    
class ComfortReward(RewardFunction):
    """Reward for occupant thermal comfort satisfaction.

    The reward is the calculated as the negative delta between the setpoint and indoor dry-bulb temperature raised to some exponent
    if outside the comfort band. If within the comfort band, the reward is the negative delta when in cooling mode and temperature
    is below the setpoint or when in heating mode and temperature is above the setpoint. The reward is 0 if within the comfort band
    and above the setpoint in cooling mode or below the setpoint and in heating mode.

    Parameters
    ----------
    env: citylearn.citylearn.CityLearnEnv
        CityLearn environment.
    band: float, default = 2.0
        Setpoint comfort band (+/-).
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    """
    
    def __init__(self, env: CityLearnEnv, band: float = None, lower_exponent: float = None, higher_exponent: float = None):
        super().__init__(env)
        self.band = band
        self.lower_exponent = lower_exponent
        self.higher_exponent = higher_exponent

    @property
    def band(self) -> float:
        return self.__band
    
    @property
    def lower_exponent(self) -> float:
        return self.__lower_exponent
    
    @property
    def higher_exponent(self) -> float:
        return self.__higher_exponent
    
    @band.setter
    def band(self, band: float):
        self.__band = 2.0 if band is None else band

    @lower_exponent.setter
    def lower_exponent(self, lower_exponent: float):
        self.__lower_exponent = 2.0 if lower_exponent is None else lower_exponent

    @higher_exponent.setter
    def higher_exponent(self, higher_exponent: float):
        self.__higher_exponent = 3.0 if higher_exponent is None else higher_exponent

    def calculate(self) -> List[float]:
        reward_list = []

        for b in self.env.buildings:
            heating = b.energy_simulation.heating_demand[b.time_step] > b.energy_simulation.cooling_demand[b.time_step]
            indoor_dry_bulb_temperature = b.energy_simulation.indoor_dry_bulb_temperature[b.time_step]
            set_point = b.energy_simulation.indoor_dry_bulb_temperature_set_point[b.time_step]
            lower_bound_comfortable_indoor_dry_bulb_temperature = set_point - self.band
            upper_bound_comfortable_indoor_dry_bulb_temperature = set_point + self.band
            delta = abs(indoor_dry_bulb_temperature - set_point)
            
            if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                exponent = self.lower_exponent if heating else self.higher_exponent
                reward = -(delta**exponent)
            
            elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                reward = 0.0 if heating else -delta

            elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                reward = -delta if heating else 0.0

            else:
                exponent = self.higher_exponent if heating else self.lower_exponent
                reward = -(delta**exponent)

            reward_list.append(reward)

        if self.env.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward
    
class SolarPenaltyAndComfortReward(RewardFunction):
    """Addition of :py:class:`citylearn.reward_function.SolarPenaltyReward` and :py:class:`citylearn.reward_function.ComfortReward`.

    Parameters
    ----------
    env: citylearn.citylearn.CityLearnEnv
        CityLearn environment.
    band: float, default = 2.0
        Setpoint comfort band (+/-).
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    coefficients: Tuple, default = (1.0, 1.0)
        Coefficents for `citylearn.reward_function.SolarPenaltyReward` and :py:class:`citylearn.reward_function.ComfortReward` values respectively.
    """
    
    def __init__(self, env: CityLearnEnv, band: float = None, lower_exponent: float = None, higher_exponent: float = None, coefficients: Tuple = None):
        super().__init__(env)
        self.__functions: List[RewardFunction] = [
            SolarPenaltyReward(env),
            ComfortReward(env, band=band, lower_exponent=lower_exponent, higher_exponent=higher_exponent)
        ]
        self.coefficients = coefficients

    @property
    def coefficients(self) -> Tuple:
        return self.__coefficients
    
    @coefficients.setter
    def coefficients(self, coefficients: Tuple):
        coefficients = [1.0]*len(self.__functions) if coefficients is None else coefficients
        assert len(coefficients) == len(self.__functions), f'{type(self).__name__} needs {len(self.__functions)} coefficients.' 
        self.__coefficients = coefficients

    def calculate(self) -> List[float]:
        reward = np.array([f.calculate() for f in self.__functions], dtype='float32')
        reward = reward*np.reshape(self.coefficients, (len(self.coefficients), 1))
        reward = reward.sum(axis=0).tolist()

        return reward