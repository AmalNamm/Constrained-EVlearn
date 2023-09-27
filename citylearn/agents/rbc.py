from typing import Any, Mapping, List
from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from citylearn.building import Building
import numpy as np
import platform

import os
import platform

class RBC(Agent):
    r"""Base rule based controller class.

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

class HourRBC(RBC):
    r"""A time-of-use rule-based controller.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    action_map: Mapping[int, float], optional
        A 24-hour action map where the key is the hour between 1-24 and the value is the action.
        For storage systems, the value is negative for discharge and positive for charge. Will
        return random actions if no map is provided.
    
    Other Parameters
    ----------------
    **kwargs: Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, env: CityLearnEnv, action_map: Mapping[int, float] = None, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.action_map = action_map

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for current time step.

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

        actions = []

        if self.action_map is None:
            actions = super().predict(observations, deterministic=deterministic)
        
        else:
            for n, o, d in zip(self.observation_names, observations, self.action_dimension):
                hour = o[n.index('hour')]
                a = [self.action_map[hour] for _ in range(d)]
                actions.append(a)

            self.actions = actions
            self.next_time_step()
        
        return actions

class BasicRBC(HourRBC):
    r"""A time-of-use rule-based controller for heat-pump charged thermal energy storage systems that charges when COP is high.

    The actions are designed such that the agent charges the controlled storage system(s) by 9.1% of its maximum capacity every
    hour between 10:00 PM and 08:00 AM, and discharges 8.0% of its maximum capacity at every other hour.

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
        action_map = {}

        for hour in Building.get_periodic_observation_metadata()['hour']:
            if 9 <= hour <= 21:
                value = -0.08
            elif (1 <= hour <= 8) or (22 <= hour <= 24):
                value = 0.091
            else:
                value = 0.0

            action_map[hour] = value

        self.action_map = action_map

class OptimizedRBC(BasicRBC):
    r"""A time-of-use rule-based controller that is an optimized version of :py:class:`citylearn.agents.rbc.BasicRBC`
    where control actions have been selected through a search grid.

    The actions are designed such that the agent discharges the controlled storage system(s) by 2.0% of its 
    maximum capacity every hour between 07:00 AM and 03:00 PM, discharges by 4.4% of its maximum capacity 
    between 04:00 PM and 06:00 PM, discharges by 2.4% of its maximum capacity between 07:00 PM and 10:00 PM, 
    charges by 3.4% of its maximum capacity between 11:00 PM to midnight and charges by 5.532% of its maximum 
    capacity at every other hour.

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
        action_map = {}

        for hour in Building.get_periodic_observation_metadata()['hour']:
            if 7 <= hour <= 15:
                value = -0.02
            elif 16 <= hour <= 18:
                value = -0.0044
            elif 19 <= hour <= 22:
                value = -0.024
            elif 23 <= hour <= 24:
                value = 0.034
            elif 1 <= hour <= 6:
                value = 0.05532
            else:
                value = 0.0

            action_map[hour] = value

        self.action_map = action_map

class BasicBatteryRBC(BasicRBC):
    r"""A time-of-use rule-based controller that is designed to take advantage of solar generation for charging.

    The actions are optimized for electrical storage (battery) such that the agent charges the controlled
    storage system(s) by 11.0% of its maximum capacity every hour between 06:00 AM and 02:00 PM, 
    and discharges 6.7% of its maximum capacity at every other hour.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    
    Other Parameters
    ----------------
    **kwargs: Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        action_map = {}

        for hour in Building.get_periodic_observation_metadata()['hour']:
            if 6 <= hour <= 14:
                value = 0.11
            else:
                value = -0.067

            action_map[hour] = value
        
        self.action_map = action_map


class V2GRBC(BasicRBC):
    r"""A time-of-use rule-based controller that is designed to take advantage of solar generation for charging and V2G.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.

    Other Parameters
    ----------------
    **kwargs: Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        action_map = {}

        for hour in Building.get_periodic_observation_metadata()['hour']:
            if 6 <= hour <= 14:
                value = 0.11
            else:
                value = -0.067

            action_map[hour] = value

        self.action_map = action_map

    def beep(self):
        """Make a beep sound based on the OS."""
        os = platform.system()
        if os == "Linux":
            print('\a')
        elif os == "Windows":
            import winsound
            winsound.Beep(440, 500)  # Beep at 440 Hz for 500 ms
        elif os == "Darwin":  # Mac OS
            import os
            os.system('say "beep"')

    from typing import List
    import numpy as np

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for current time step."""

        actions = []

        if self.action_map is None:
            actions = super().predict(observations, deterministic=deterministic)
        else:
            for n, o, d in zip(self.observation_names, observations, self.action_dimension):
                hour = o[n.index('hour')]
                a = [self.action_map[hour] for _ in range(d)]
                actions.append(a)

        for i, b in enumerate(self.env.buildings):
            if b.chargers:
                for charger_index, charger in reversed(list(enumerate(b.chargers))):
                    # If no EV is connected, set action to 0
                    if not charger.connected_ev:
                        actions[i][-charger_index - 1] = 0.001
                        continue
                    else:
                        # Calculate the time left until departure
                        if charger.connected_ev.ev_simulation.estimated_departure_time[b.time_step] >= 0:
                            hours_until_departure = charger.connected_ev.ev_simulation.estimated_departure_time[b.time_step]
                        else:
                            hours_until_departure = 24


                        # Calculate the SOC deficit or surplus
                        required_soc = charger.connected_ev.ev_simulation.required_soc_departure[b.time_step] #%
                        actual_soc = (charger.connected_ev.battery.soc[-1] * 100) / charger.connected_ev.battery.capacity #%
                        soc_diff = required_soc - actual_soc #%

                        # Calculate net demand: non_shiftable_load_demand - solar generation
                        #net_demand = b.non_shiftable_load_demand[b.time_step] - b.solar_generation[b.time_step]
                        net_demand = b.net_electricity_consumption_without_storage[b.time_step]
                        net_demand_next = b.energy_simulation.non_shiftable_load[b.time_step+1] + b.energy_simulation.solar_generation[b.time_step]

                        # TOU pricing condition
                        current_price = b.pricing.electricity_pricing[b.time_step]
                        future_price_avg = np.mean(
                            b.pricing.electricity_pricing[b.time_step:b.time_step + hours_until_departure])



                        # Highest Priority: Address SOC deficit but distribute the charging over remaining hours


                        if soc_diff > 5 and hours_until_departure <= 2:
                            # Calculate the proportional SOC deficit
                            proportional_soc_diff = soc_diff / hours_until_departure  # How much i have to charge in % by hour
                            # This gives a charging rate that aims to fill the deficit evenly over remaining hours
                            target_charging_rate = (proportional_soc_diff / 100) * charger.connected_ev.battery.capacity  # KWH

                            desired_charging_power = min(target_charging_rate, (soc_diff/100)*charger.connected_ev.battery.capacity,
                                                         charger.max_charging_power)  # Use 75% of the surplus energy
                            actions[i][-charger_index - 1] = min(desired_charging_power / charger.max_charging_power, 1)

                        elif soc_diff > 50 and  hours_until_departure > 3:
                            # Calculate the proportional SOC deficit
                            proportional_soc_diff = soc_diff / hours_until_departure  # How much i have to charge in % by hour
                            # This gives a charging rate that aims to fill the deficit evenly over remaining hours
                            target_charging_rate = (proportional_soc_diff / 100) * charger.connected_ev.battery.capacity  # KWH


                            # Aim to fill the deficit evenly over the remaining hours
                            actions[i][-charger_index - 1] = target_charging_rate / charger.max_charging_power

                        # Medium Priority: Help the grid if there's high demand and SOC needs are met
                        elif net_demand_next > 0 and hours_until_departure > 3 and current_price > future_price_avg:

                            desired_discharging_power = max(-net_demand,
                                                            -charger.max_discharging_power)  # We're discharging half of the net demand to balance
                            actions[i][-charger_index - 1] = max(desired_discharging_power / charger.max_discharging_power, -1)

                        # Next, charge based on available surplus, but not at the max rate
                        elif net_demand_next < 0:
                            desired_charging_power = min(-net_demand,
                                                         charger.max_charging_power)  # Use 75% of the surplus energy
                            actions[i][-charger_index - 1] = min(desired_charging_power / charger.max_charging_power, 1)

                        # Lower Priority: If there's time and the current price of electricity is favorable
                        elif hours_until_departure > 3 and current_price < future_price_avg:
                            desired_charging_power = min((soc_diff/100)*charger.connected_ev.battery.capacity,
                                                         charger.max_charging_power)  # Use 75% of the surplus energy
                            actions[i][-charger_index - 1] = min(desired_charging_power / charger.max_charging_power, 1)

                        else:
                            actions[i][-charger_index - 1] = 0.01

        self.actions = actions
        self.next_time_step()

        actions_modified = []
        for action_list in actions:
            modified_list = []
            for x in action_list:
                if x == 0:
                    print("Encountered a 0 in actions!")
                    #self.beep()
                    modified_list.append(0.001)
                else:
                    modified_list.append(x)
            actions_modified.append(modified_list)

        actions = actions_modified

        return actions





