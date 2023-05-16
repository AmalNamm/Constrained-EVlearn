import inspect
import math
from typing import List, Mapping, Tuple, Union
from gym import spaces
import numpy as np
from citylearn.base import Environment
from citylearn.data import EnergySimulation, CarbonIntensity, Pricing, Weather
from citylearn.energy_model import Battery, ElectricHeater, HeatPump, PV, StorageTank, Charger
from citylearn.preprocessing import Normalize, PeriodicNormalization

class EV(Environment):
    def __init__(self, battery: Battery, energy_consumption_rate: float, time_step: float = 15):
        """
        Initialize the EVCar class.

        Parameters
        ----------
        battery : Battery
            An instance of the Battery class.
        energy_consumption_rate : float
            Energy consumption rate of the car while driving in kWh per distance unit (e.g., kWh/k).
        time_step : float, default: 15
            Time step duration in minutes.
        """
        self.battery = battery
        self.energy_consumption_rate = energy_consumption_rate
        self.location = "parked_not_charging"
        self.time_step = time_step
        self.distance_travelled = 0

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
        distance = speed * (self.time_step / 60)  # Convert speed to distance based on the time step.
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
    def location(self) -> str:
        """Return the car's location status."""
        return self._location

    @location.setter
    def location(self, location: str):
        if location not in ["charging", "parked_not_charging", "travelling"]:
            raise ValueError("Invalid location status.")
        self._location = location

    @property
    def energy_consumption_rate(self) -> float:
        """Return the energy consumption rate of the car while driving."""
        return self._energy_consumption_rate

    @energy_consumption_rate.setter
    def energy_consumption_rate(self, energy_consumption_rate: float):
        if energy_consumption_rate < 0:
            raise ValueError("Energy consumption rate must be non-negative.")
        self._energy_consumption_rate = energy_consumption_rate

    @property
    def time_step(self) -> float:
        """Return the time step duration."""
        return self._time_step

    @time_step.setter
    def time_step(self, time_step: float):
        if time_step <= 0:
            raise ValueError("Time step duration must be greater than zero.")
        self._time_step = time_step

    def reset(self):
        """
        Reset the EVCar to its initial state.
        """
        self.location = "parked_not_charging"
        self.distance_travelled = 0
        self.battery.reset()