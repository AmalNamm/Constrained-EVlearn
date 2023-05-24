import inspect
import math
from typing import List, Mapping, Tuple, Union
from gym import spaces
import numpy as np
from citylearn.base import Environment
from citylearn.data import EnergySimulation, CarbonIntensity, Pricing, Weather, EVSimulation
from citylearn.energy_model import Battery, ElectricHeater, HeatPump, PV, StorageTank, Charger
from citylearn.preprocessing import Normalize, PeriodicNormalization

class EV(Environment):

    def __init__(self, ev_simulation: EVSimulation, energy_consumption_rate: float, location: str,  observation_metadata: Mapping[str, bool],
    action_metadata: Mapping[str, bool], battery: Battery = None, name: str = None, ** kwargs):
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
        location : str, restricted to either "charging", "parked_not_charging", "travelling"
            Location of the EV
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
        self.location = location or "parked_not_charging"
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.distance_travelled = 0

        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key:value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))
        }
        super().__init__(**kwargs)

    def travel(self, speed: float):
        """
        Update the car's location to 'travelling', calculate the energy consumption based on the
        provided speed, and update the battery's state of charge.

        Parameters
        ----------
        speed : float
            The car's speed in distance units per hour (e.g., km/h or miles/h).
        """
        self.location = "travelling"
        distance = speed * super.seconds_per_time_step  # Convert speed to distance based on the time step. TODO Min or seconds ?
        self.distance_travelled += distance
        energy_consumption = distance * self.energy_consumption_rate
        self.battery.charge(-energy_consumption)  # Discharge the battery to account for energy consumption while driving.

    def park(self):
        """Update the car's location to 'parked_not_charging'."""
        self.location = "parked_not_charging"

    def charge(self, energy: float):
        """
        Charge or discharge the battery with the specified amount of energy.

        Parameters
        ----------
        energy : float
            Energy to charge if (+) or discharge if (-) in kWh.
        """
        self.battery.charge(energy)

    @property
    def ev_simulation(self) -> str:
        """Return the EV simulation data."""
        return self.__ev_simulation

    @ev_simulation.setter
    def ev_simulation(self, ev_simulation: EVSimulation):
        self.__ev_simulation = ev_simulation

    @property
    def location(self) -> str:
        """Return the car's location status."""
        return self.__location

    @location.setter
    def location(self, location: str):
        if location not in ["charging", "parked_not_charging", "travelling"]:
            raise ValueError("Invalid location status.")
        self.__location = location

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

    @property
    def battery(self) -> Battery:
        """Battery for EV."""
        return self.__battery

    @battery.setter
    def battery(self, battery: Battery):
        self.__battery = Battery(0.0) if battery is None else battery



    def reset(self):
        """
        Reset the EVCar to its initial state.
        """
        super().reset()
        self.location = "parked_not_charging"
        self.distance_travelled = 0
        self.battery.reset()

    def estimate_observation_space(self):#TODO
        pass

    def estimate_action_space(self): #TODO
        pass

    def autosize_battery(self, **kwargs):
        """Autosize `Battery` for a typical EV.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `electrical_storage` `autosize` function.
        """

        self.battery.autosize_for_EV()
