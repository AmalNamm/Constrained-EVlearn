import inspect
import math
from typing import List, Mapping, Tuple, Union, Dict
from gym import spaces
import numpy as np
from citylearn.base import Environment
from citylearn.data import EnergySimulation, CarbonIntensity, Pricing, Weather, EVSimulation
from citylearn.energy_model import Battery, ElectricHeater, HeatPump, PV, StorageTank
from citylearn.preprocessing import Normalize, PeriodicNormalization
import random

class EV(Environment):

    def __init__(self, ev_simulation: EVSimulation, energy_consumption_rate: float,
                 observation_metadata: Mapping[str, bool],
                 action_metadata: Mapping[str, bool], battery: Battery = None,
                 image_path: str = None, name: str = None, **kwargs):
        """
        Initialize the EVCar class.

        Parameters
        ----------
        ev_simulation : EVSimulation
            Temporal features, locations, predicted SOCs and more.
        battery : Battery
            An instance of the Battery class.
        energy_consumption_rate : float
            Energy consumption rate of the car while driving in kWh per distance unit (e.g., kWh/k).
        observation_metadata : dict
            Mapping of active and inactive observations.
        action_metadata : dict
            Mapping od active and inactive actions.
        name : str, optional
            Unique EV name.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        self.ev_simulation = ev_simulation
        self.name = name
        self.battery = battery
        self.energy_consumption_rate = energy_consumption_rate
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.non_periodic_normalized_observation_space_limits = None
        self.periodic_normalized_observation_space_limits = None
        self.observation_space = self.estimate_observation_space()
        self.action_space = self.estimate_action_space()
        self.image_path = image_path
        self.__observation_epsilon = 0.0  # to avoid out of bound observations


        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key: value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))
        }
        super().__init__(**kwargs)

    @property
    def ev_simulation(self) -> EVSimulation:
        """Return the EV simulation data."""
        return self.__ev_simulation

    @ev_simulation.setter
    def ev_simulation(self, ev_simulation: EVSimulation):
        self.__ev_simulation = ev_simulation

    @property
    def energy_consumption_rate(self) -> float:
        """Return the energy consumption rate of the car while driving."""
        return self.__energy_consumption_rate

    @energy_consumption_rate.setter
    def energy_consumption_rate(self, energy_consumption_rate: float):
        if energy_consumption_rate < 0:
            raise ValueError("Energy consumption rate must be non-negative.")
        self.__energy_consumption_rate = energy_consumption_rate

    @property
    def name(self) -> str:
        """Unique building name."""

        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def image_path(self) -> str:
        """Unique building name."""

        return self.__image_path

    @image_path.setter
    def image_path(self, image_path: str):
        self.__image_path = image_path

    @property
    def observation_metadata(self) -> Mapping[str, bool]:
        """Mapping of active and inactive observations."""

        return self.__observation_metadata

    @property
    def action_metadata(self) -> Mapping[str, bool]:
        """Mapping od active and inactive actions."""

        return self.__action_metadata

    @observation_metadata.setter
    def observation_metadata(self, observation_metadata: Mapping[str, bool]):
        self.__observation_metadata = observation_metadata

    @action_metadata.setter
    def action_metadata(self, action_metadata: Mapping[str, bool]):
        self.__action_metadata = action_metadata

    @property #TODO initilizar com soc init
    def battery(self) -> Battery:
        """Battery for EV."""
        return self.__battery

    @battery.setter
    def battery(self, battery: Battery):
        self.__battery = Battery(0.0, 0.0) if battery is None else battery

    @property
    def observation_space(self) -> spaces.Box:
        """Agent observation space."""

        return self.__observation_space

    @property
    def action_space(self) -> spaces.Box:
        """Agent action spaces."""

        return self.__action_space

    @property
    def active_observations(self) -> List[str]:
        """Observations in `observation_metadata` with True value i.e. obeservable."""

        return [k for k, v in self.observation_metadata.items() if v]

    @property
    def active_actions(self) -> List[str]:
        """Actions in `action_metadata` with True value i.e.
        indicates which storage systems are to be controlled during simulation."""

        return [k for k, v in self.action_metadata.items() if v]

    @observation_space.setter
    def observation_space(self, observation_space: spaces.Box):
        self.__observation_space = observation_space
        self.non_periodic_normalized_observation_space_limits = self.estimate_observation_space_limits(
            include_all=True, periodic_normalization=False
        )
        self.periodic_normalized_observation_space_limits = self.estimate_observation_space_limits(
            include_all=True, periodic_normalization=True
        )

    @action_space.setter
    def action_space(self, action_space: spaces.Box):
        self.__action_space = action_space

    @property #TODO
    def energy_from_cooling_device_to_cooling_storage(self) -> np.ndarray:
        """Energy supply from `cooling_device` to `cooling_storage` time series, in [kWh]."""

        return np.array(self.cooling_storage.energy_balance, dtype=float).clip(min=0)


    #def travel(self):
    #    print("The EV is now in transit ")
    #    """
    #    Calculate the energy consumption based on a randomly
    #    generated speed, and update the battery's state of charge using the `charge` method.
#
    #    """
    #    # Generate a realistic speed randomly (between 30km/h to 120km/h seems realistic)
    #    speed = random.uniform(30, 120)  # in km/hr
#
    #    # Convert speed to distance based on the time step (assuming that self.seconds_per_time_step is in seconds)
    #    distance = speed * (self.seconds_per_time_step / 3600)  # Convert km/hr to km/second
#
    #    # Calculate the energy consumption
    #    energy_consumption = distance * self.energy_consumption_rate
#
    #    # Introduce variability in energy consumption and charging
    #    if random.random() < 0.9:  # 90% chance to discharge while in transit
    #        # Discharge the battery to account for energy consumption while driving.
    #        print("Discharging in transit")
    #        self.battery.charge(-energy_consumption)
    #    else:  # 10% chance to charge while at another location
    #        # Generate a random charging speed in kWh (between 7-120 kWh seems realistic for 1 hour)
    #        charging_speed = random.uniform(7, 120)  # in kWh per hour
    #        # Calculate the amount of energy charged based on the time step
    #        energy_charged = charging_speed * (self.seconds_per_time_step / 3600)  # Convert kWh/hr to kWh/second
    #        # Charge the battery at another location while travelling
    #        print(f"Charging in transit {energy_charged}")
    #        self.battery.charge(energy_charged)

    #def park(self):
    #    print("The EV is parking and it is not plugged in")
    #    """Update the car's location to 'parked_not_charging'."""
    #    # Simulate power usage while parked by decreasing the state of charge slightly
    #    # Assume a small amount of power loss, e.g., between 0 and 0.05 kWh
    #    power_loss_while_parked = random.uniform(0, 0.005)  # in kWh
    #    self.battery.charge(-power_loss_while_parked)


    def adjust_ev_soc_on_system_connection(self, soc_system_connection):
        """
        Adjusts the state of charge (SoC) of an electric vehicle's (EV's) battery upon connection to the system.

        When an EV is in transit, the system "loses" the connection and does not know how much battery
        has been used during travel. As such, when an EV enters an incoming or connected state, its battery
        SoC is updated to be close to the predicted SoC at arrival present in the EV dataset.

        However, predictions sometimes fail, so this method introduces variability for the simulation by
        randomly creating a discrepancy between the predicted value and a "real-world inspired" value. This discrepancy
        is generated using a normal (Gaussian) distribution, which is more likely to produce values near 0 and less
        likely to produce extreme values.

        The range of potential variation is between -30% to +30% of the predicted SoC, with most of the values
        being close to 0 (i.e., the prediction). The exact amount of variation is calculated by taking a random
        value from the normal distribution and scaling it by the predicted SoC. This value is then added to the
        predicted SoC to get the actual SoC, which can be higher or lower than the prediction.

        The difference between the actual SoC and the initial SoC (before the adjustment) is passed to the
        battery's charge method. If the difference is positive, the battery is charged; if the difference is negative,
        the battery is discharged.

        For example, if the EV dataset has a predicted SoC at arrival of 20% (of the battery's total capacity),
        this method can randomly adjust the EV's battery to 22% or 19%, or even by a larger margin such as 40%.

        Args:
        soc_system_connection (float): The predicted SoC at system connection, expressed as a percentage of the
        battery's total capacity.
        """

        # Get the initial SoC in kWh from the battery
        soc_init_kwh = self.battery.soc_init

        # Calculate the system connection SoC in kWh
        soc_system_connection_kwh = self.battery.capacity * (soc_system_connection / 100)

        # Determine the range for random variation.
        # Here we use a normal distribution centered at 0 and a standard deviation of 0.1.
        # We also make sure that the values are truncated at -30% and +30%.
        variation_percentage = np.clip(np.random.normal(0, 0.1), -0.3, 0.3)

        # Apply the variation
        variation_kwh = variation_percentage * soc_system_connection_kwh

        # Calculate the final SoC in kWh
        soc_final_kwh = soc_system_connection_kwh + variation_kwh

        # Charge or discharge the battery to the new SoC.
        self.battery.charge(soc_final_kwh - soc_init_kwh)

    def next_time_step(self) -> Mapping[int, str]:

        """
        Advance EV to the next `time_step` by
        """

        self.battery.next_time_step()
        super().next_time_step()
        self.update_variables() #TODO Might need to go below the logic of charging

        if self.ev_simulation.ev_state[self.time_step] == 1 or self.ev_simulation.ev_state[self.time_step] == 2: #connected or incoming
            self.adjust_ev_soc_on_system_connection(self.ev_simulation.estimated_soc_arrival[self.time_step])

    def reset(self): #TODO
        """
        Reset the EVCar to its initial state.
        """
        super().reset()

        #object reset
        self.battery.reset()

        # variable reset
        #self.__cooling_electricity_consumption = []
        #self.__heating_electricity_consumption = []
        #self.__dhw_electricity_consumption = []
        #self.__solar_generation = self.pv.get_generation(self.energy_simulation.solar_generation) * -1
        #self.__net_electricity_consumption = []
        #self.__net_electricity_consumption_emission = []
        #self.__net_electricity_consumption_cost = []
        self.update_variables()

        ## reset controlled variables
        #self.energy_simulation.cooling_demand = self.__cooling_demand_without_partial_load.copy()
        #self.energy_simulation.heating_demand = self.__heating_demand_without_partial_load.copy()
        #self.energy_simulation.indoor_dry_bulb_temperature = self.__indoor_dry_bulb_temperature_without_partial_load.copy()

    def update_variables(self): #TODO
        """Update cooling, heating, dhw and net electricity consumption as well as net electricity consumption cost and carbon emissions."""
        pass
       ## cooling electricity consumption
       #cooling_demand = self.energy_simulation.cooling_demand[self.time_step] + self.cooling_storage.energy_balance[
       #    self.time_step]
       #cooling_consumption = self.cooling_device.get_input_power(cooling_demand,
       #                                                          self.weather.outdoor_dry_bulb_temperature[
       #                                                              self.time_step], heating=False)
       #self.__cooling_electricity_consumption.append(cooling_consumption)

       ## heating electricity consumption
       #heating_demand = self.energy_simulation.heating_demand[self.time_step] + self.heating_storage.energy_balance[
       #    self.time_step]

       #if isinstance(self.heating_device, HeatPump):
       #    heating_consumption = self.heating_device.get_input_power(heating_demand,
       #                                                              self.weather.outdoor_dry_bulb_temperature[
       #                                                                  self.time_step], heating=True)
       #else:
       #    heating_consumption = self.dhw_device.get_input_power(heating_demand)

       #self.__heating_electricity_consumption.append(heating_consumption)

       ## dhw electricity consumption
       #dhw_demand = self.energy_simulation.dhw_demand[self.time_step] + self.dhw_storage.energy_balance[self.time_step]

       #if isinstance(self.dhw_device, HeatPump):
       #    dhw_consumption = self.dhw_device.get_input_power(dhw_demand,
       #                                                      self.weather.outdoor_dry_bulb_temperature[self.time_step],
       #                                                      heating=True)
       #else:
       #    dhw_consumption = self.dhw_device.get_input_power(dhw_demand)

       #self.__dhw_electricity_consumption.append(dhw_consumption)

       ## net electricity consumption
       #net_electricity_consumption = cooling_consumption \
       #                              + heating_consumption \
       #                              + dhw_consumption \
       #                              + self.electrical_storage.electricity_consumption[self.time_step] \
       #                              + self.energy_simulation.non_shiftable_load[self.time_step] \
       #                              + self.__solar_generation[self.time_step]
       #self.__net_electricity_consumption.append(net_electricity_consumption)

       ## net electriciy consumption cost
       #self.__net_electricity_consumption_cost.append(
       #    net_electricity_consumption * self.pricing.electricity_pricing[self.time_step])

       ## net electriciy consumption emission
       #self.__net_electricity_consumption_emission.append(
       #    max(0, net_electricity_consumption * self.carbon_intensity.carbon_intensity[self.time_step]))

    def observations(self, include_all: bool = None, normalize: bool = None, periodic_normalization: bool = None) -> \
            Mapping[str, float]:
        r"""Observations at current time step.

        Parameters
        ----------
        include_all: bool, default: False,
            Whether to estimate for all observations as listed in `observation_metadata` or only those that are active.
        normalize : bool, default: False
            Whether to apply min-max normalization bounded between [0, 1].
        periodic_normalization: bool, default: False
            Whether to apply sine-cosine normalization to cyclic observations.

        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.
        """

        normalize = False if normalize is None else normalize
        periodic_normalization = False if periodic_normalization is None else periodic_normalization
        include_all = False if include_all is None else include_all

        ev_data = vars(self.ev_simulation)
        unwanted_keys = ['month', 'hour', 'day_type', "charger"]

        data = {
            **{k: v[self.time_step] for k, v in ev_data.items() if k not in unwanted_keys},
            'ev_soc': self.battery.soc[self.time_step] / self.battery.capacity #TODO not working for some reason
        }

        if include_all:
            valid_observations = list(self.observation_metadata.keys())
        else:
            valid_observations = self.active_observations

        observations = {k: data[k] for k in valid_observations if k in data.keys()}
        unknown_observations = list(set(valid_observations).difference(observations.keys()))
        assert len(unknown_observations) == 0, f'Unknown observations: {unknown_observations}'

        low_limit, high_limit = self.periodic_normalized_observation_space_limits
        periodic_observations = self.get_periodic_observation_metadata()

        if periodic_normalization:
            observations_copy = {k: v for k, v in observations.items()}
            observations = {}
            pn = PeriodicNormalization(x_max=0)

            for k, v in observations_copy.items():
                if k in periodic_observations:
                    pn.x_max = max(periodic_observations[k])
                    sin_x, cos_x = v * pn
                    observations[f'{k}_cos'] = cos_x
                    observations[f'{k}_sin'] = sin_x
                else:
                    observations[k] = v
        else:
            pass

        if normalize:
            nm = Normalize(0.0, 1.0)

            for k, v in observations.items():
                nm.x_min = low_limit[k]
                nm.x_max = high_limit[k]
                observations[k] = v * nm
        else:
            pass

        return observations

    @staticmethod
    def get_periodic_observation_metadata() -> dict[str, range]:
        r"""Get periodic observation names and their minimum and maximum values for periodic/cyclic normalization.

        Returns
        -------
        periodic_observation_metadata: Mapping[str, int]
            Observation low and high limits.
        """

        return {
            'hour': range(1, 25),
            'day_type': range(1, 9),
            'month': range(1, 13)
        }

    def apply_actions(self,
                      connect_to_charger_action: int = None, disconnect_from_charger_action: int = None,
                      charging_amount_action: float = None):
        r"""Update EV connection and charging status for the next timestep.
        Parameters
        ----------
        connect_to_charger_action : int, default: np.nan
            ID of the charger to connect the EV to. Use np.nan or a negative number for no connection.
        disconnect_from_charger_action : int, default: np.nan
            ID of the charger to disconnect the EV from. Use np.nan or a negative number for no disconnection.
        charging_amount_action : float, default: 0.0
            Fraction of `EV_battery` `capacity` to charge by.
        """
        connect_to_charger_action = -1 if connect_to_charger_action is None or math.isnan(
            connect_to_charger_action) else connect_to_charger_action
        disconnect_from_charger_action = -1 if disconnect_from_charger_action is None or math.isnan(
            disconnect_from_charger_action) else disconnect_from_charger_action
        charging_amount_action = 0.0 if charging_amount_action is None or math.isnan(
            charging_amount_action) else charging_amount_action
        if connect_to_charger_action >= 0:
            self.connect_EV_to_charger(connect_to_charger_action)
        if disconnect_from_charger_action >= 0:
            self.disconnect_EV_from_charger(disconnect_from_charger_action)
        self.update_EV_charging(charging_amount_action)

    def estimate_observation_space(self, include_all: bool = None, normalize: bool = None,
                                   periodic_normalization: bool = None) -> spaces.Box:  # TODO
        r"""Get estimate of observation spaces.
        Parameters
        ----------
        include_all: bool, default: False,
            Whether to estimate for all observations as listed in `observation_metadata` or only those that are active.
        normalize : bool, default: False
            Whether to apply min-max normalization bounded between [0, 1].
        periodic_normalization: bool, default: False
            Whether to apply sine-cosine normalization to cyclic observations including hour, day_type and month.
        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.
        """
        normalize = False if normalize is None else normalize
        normalized_observation_space_limits = self.estimate_observation_space_limits(
            include_all=include_all, periodic_normalization=True
        )
        unnormalized_observation_space_limits = self.estimate_observation_space_limits(
            include_all=include_all, periodic_normalization=False
        )
        if normalize:
            low_limit, high_limit = normalized_observation_space_limits
            low_limit = [0.0] * len(low_limit)
            high_limit = [1.0] * len(high_limit)
        else:
            low_limit, high_limit = unnormalized_observation_space_limits
            low_limit = list(low_limit.values())
            high_limit = list(high_limit.values())
        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    def estimate_observation_space_limits(self, include_all: bool = None, periodic_normalization: bool = None) -> Tuple[
        Mapping[str, float], Mapping[str, float]]:  # TODO
        r"""Get estimate of observation space limits.
        Find minimum and maximum possible values of all the observations, which can then be used by the RL agent to scale the observations and train any function approximators more effectively.
        Parameters
        ----------
        include_all: bool, default: False,
            Whether to estimate for all observations as listed in `observation_metadata` or only those that are active.
        periodic_normalization: bool, default: False
            Whether to apply sine-cosine normalization to cyclic observations including hour, day_type and month.
        Returns
        -------
        observation_space_limits : Tuple[Mapping[str, float], Mapping[str, float]]
            Observation low and high limits.
        Notes
        -----
        Lower and upper bounds of net electricity consumption are rough estimates and may not be completely accurate hence,
        scaling this observation-variable using these bounds may result in normalized values above 1 or below 0.
        """
        include_all = False if include_all is None else include_all
        observation_names = list(self.observation_metadata.keys()) if include_all else self.active_observations
        periodic_normalization = False if periodic_normalization is None else periodic_normalization
        periodic_observations = self.get_periodic_observation_metadata()
        low_limit, high_limit = {}, {}
        for key in observation_names:
            if key == 'ev_state':
                low_limit[key] = 0
                high_limit[key] = 1
            if key == 'charger':
                    low_limit[key] = 0
                    high_limit[key] = 7  #
            elif key in "estimated_departure_time" or key in "estimated_arrival_time":
                    low_limit[key] = 0
                    high_limit[key] = 24
            elif key in "required_soc_departure" or key in "estimated_soc_arrival"  or key in "ev_soc":
                    low_limit[key] = 0.0
                    high_limit[key] = 1.0
        low_limit = {k: v - 0.05 for k, v in low_limit.items()}
        high_limit = {k: v + 0.05 for k, v in high_limit.items()}
        return low_limit, high_limit

    def estimate_action_space(self) -> spaces.Box:
        r"""Get estimate of action spaces.
        Find minimum and maximum possible values of all the actions, which can then be used by the RL agent to scale the selected actions.
        Returns
        -------
        action_space : spaces.Box
            Action low and high limits.
        Notes
        -----
        The lower and upper bounds for the `cooling_storage`, `heating_storage` and `dhw_storage` actions are set to (+/-) 1/maximum_demand for each respective end use,
        as the energy storage device can't provide the building with more energy than it will ever need for a given time step. .
        For example, if `cooling_storage` capacity is 20 kWh and the maximum `cooling_demand` is 5 kWh, its actions will be bounded between -5/20 and 5/20.
        These boundaries should speed up the learning process of the agents and make them more stable compared to setting them to -1 and 1.
        """
        low_limit, high_limit = [], []
        for key in self.active_actions:
            if key == 'ev_storage':
                limit = self.battery.nominal_power / self.battery.capacity
                low_limit.append(-limit)
                high_limit.append(limit)
            elif key == "ev_connection_action":  # TODO might change
                low_limit.append(0)
                high_limit.append(7)  # number of chargers
            elif key == "ev_disconnection_action":  # TODO might change
                low_limit.append(0)
                high_limit.append(3)  # number of evs
        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    def autosize_battery(self, **kwargs):
        """Autosize `Battery` for a typical EV.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `electrical_storage` `autosize` function.
        """

        self.battery.autosize_for_EV()

    @staticmethod
    def observations_length() -> Mapping[str, int]:
        r"""Get periodic observation names and their minimum and maximum values for periodic/cyclic normalization.

        Returns
        -------
        periodic_observation_metadata: Mapping[str, int]
            Observation low and high limits.
        """

        return {
            'hour': range(1, 25),
            'day_type': range(1, 9),
            'month': range(1, 13)
        }

    def __str__(self):
        ev_simulation_attrs = [
            f"EV simulation (time_step={self.time_step}):",
            f"Month: {self.ev_simulation.month[self.time_step]}",
            f"Hour: {self.ev_simulation.hour[self.time_step]}",
            f"Day Type: {self.ev_simulation.day_type[self.time_step]}",
            f"State: {self.ev_simulation.ev_state[self.time_step]}",
            f"Estimated Departure Time: {self.ev_simulation.estimated_departure_time[self.time_step]}",
            f"Required Soc At Departure: {self.ev_simulation.required_soc_departure[self.time_step]}",
            f"Estimated Arrival Time: {self.ev_simulation.estimated_arrival_time[self.time_step]}",
            f"Estimated Soc Arrival: {self.ev_simulation.estimated_soc_arrival[self.time_step]}"
        ]

        ev_simulation_str = '\n'.join(ev_simulation_attrs)

        return (
            f"EV {self.name}:\n"
            f"  Energy consumption rate: {self.energy_consumption_rate}\n"
            f"  Battery: {self.battery}\n\n"
            f"Simulation details:\n"
            f"  {ev_simulation_str}"
        )

