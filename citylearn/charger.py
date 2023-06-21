from typing import List, Dict

from citylearn import EV

ZERO_DIVISION_CAPACITY = 0.00001
class Charger():
    def __init__(
            self,
            nominal_power: float,
            efficiency: float = None,
            charger_id: str = None,
            charger_type: int = None,
            max_charging_power: float = 50.0,
            min_charging_power: float = 0.0,
            max_discharging_power: float = 50.0,
            min_discharging_power: float = 0.0,
            charge_efficiency_curve: Dict[float, float] = None,
            discharge_efficiency_curve: Dict[float, float] = None,
            image_path: str = None,
            connected_ev: EV = None, incoming_ev: EV = None
    ):
        r"""Initializes the `Electric Vehicle Charger` class with the given attributes.

        Parameters
        ----------
        charger_id: str
            Id through which the charger is uniquely identified in the system
        charger_type: int
            Either private (0) or public (1) charger
        max_charging_power : float, default 50
            Maximum charging power in kW.
        min_charging_power : float, default 0
            Minimum charging power in kW.
        max_discharging_power : float, default 50
            Maximum discharging power in kW.
        min_discharging_power : float, default 0
            Minimum discharging power in kW.
        charge_efficiency_curve : dict, default {3.6: 0.95, 7.2: 0.97, 22: 0.98, 50: 0.98}
            Efficiency curve for charging containing power levels and corresponding efficiency values.
        discharge_efficiency_curve : dict, default {3.6: 0.95, 7.2: 0.97, 22: 0.98, 50: 0.98}
            Efficiency curve for discharging containing power levels and corresponding efficiency values.
        max_connected_cars : int, default 1
            Maximum number of cars that can be connected to the charger simultaneously.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super classes.
        """

        self.nominal_power = nominal_power
        self.efficiency = efficiency
        self.charger_id = charger_id
        self.charger_type = charger_type
        self.max_charging_power = max_charging_power
        self.min_charging_power = min_charging_power
        self.max_discharging_power = max_discharging_power
        self.min_discharging_power = min_discharging_power
        self.charge_efficiency_curve = charge_efficiency_curve or {3.6: 0.95, 7.2: 0.97, 22: 0.98, 50: 0.98}
        self.discharge_efficiency_curve = discharge_efficiency_curve or {3.6: 0.95, 7.2: 0.97, 22: 0.98, 50: 0.98}
        self.image_path = image_path
        self.connected_ev = connected_ev or None
        self.incoming_ev = incoming_ev or None

    @property
    def charger_id(self) -> str:
        """ID of the charger."""
        return self.__charger_id

    @property
    def charger_type(self) -> int:
        """Type of the charger."""
        return self.__charger_type

    @property
    def max_charging_power(self) -> float:
        """Maximum charging power in kW."""
        return self.__max_charging_power

    @property
    def image_path(self) -> str:
        """Unique building name."""

        return self.__image_path

    @image_path.setter
    def image_path(self, image_path: str):
        self.__image_path = image_path

    @property
    def min_charging_power(self) -> float:
        """Minimum charging power in kW."""
        return self.__min_charging_power

    @property
    def max_discharging_power(self) -> float:
        """Maximum discharging power in kW."""
        return self.__max_discharging_power

    @property
    def min_discharging_power(self) -> float:
        """Minimum discharging power in kW."""
        return self.__min_discharging_power

    @property
    def charge_efficiency_curve(self) -> dict:
        """Efficiency curve for charging containing power levels and corresponding efficiency values."""
        return self.__charge_efficiency_curve

    @property
    def discharge_efficiency_curve(self) -> dict:
        """Efficiency curve for discharging containing power levels and corresponding efficiency values."""
        return self.__discharge_efficiency_curve

    @property
    def connected_ev(self) -> EV:
        """EV currently connected to charger"""
        return self.__connected_ev

    @property
    def incoming_ev(self) -> EV:
        """EV incoming to charger"""
        return self.__incoming_ev

    @charger_id.setter
    def charger_id(self, charger_id: str):
        self.__charger_id = charger_id

    @property
    def efficiency(self) -> float:
        """Technical efficiency."""

        return self.__efficiency

    @efficiency.setter
    def efficiency(self, efficiency: float):
        if efficiency is None:
            self.__efficiency = 1.0
        else:
            assert efficiency > 0, 'efficiency must be > 0.'
            self.__efficiency = efficiency



    @property
    def nominal_power(self) -> float:
        r"""Nominal power."""

        return self.__nominal_power

    @property
    def electricity_consumption(self) -> List[float]:
        r"""Electricity consumption time series."""

        return self.__electricity_consumption

    @property
    def available_nominal_power(self) -> float:
        r"""Difference between `nominal_power` and `electricity_consumption` at current `time_step`."""

        return None if self.nominal_power is None else self.nominal_power - self.electricity_consumption[self.time_step]

    @nominal_power.setter
    def nominal_power(self, nominal_power: float):
        if nominal_power is None or nominal_power == 0:
            self.__nominal_power = ZERO_DIVISION_CAPACITY
        else:
            assert nominal_power >= 0, 'nominal_power must be >= 0.'
            self.__nominal_power = nominal_power

    def update_electricity_consumption(self, electricity_consumption: float):
        r"""Updates `electricity_consumption` at current `time_step`.

        Parameters
        ----------
        electricity_consumption : float
            value to add to current `time_step` `electricity_consumption`. Must be >= 0.
        """

        assert electricity_consumption >= 0, 'electricity_consumption must be >= 0.'
        self.__electricity_consumption[self.time_step] += electricity_consumption

    @charger_type.setter
    def charger_type(self, charger_type: str):
        self.__charger_type = charger_type

    @max_charging_power.setter
    def max_charging_power(self, max_charging_power: float):
        self.__max_charging_power = max_charging_power

    @min_charging_power.setter
    def min_charging_power(self, min_charging_power: float):
        self.__min_charging_power = min_charging_power

    @max_discharging_power.setter
    def max_discharging_power(self, max_discharging_power: float):
        self.__max_discharging_power = max_discharging_power

    @min_discharging_power.setter
    def min_discharging_power(self, min_discharging_power: float):
        self.__min_discharging_power = min_discharging_power

    @charge_efficiency_curve.setter
    def charge_efficiency_curve(self, charge_efficiency_curve: dict):
        self.__charge_efficiency_curve = charge_efficiency_curve

    @discharge_efficiency_curve.setter
    def discharge_efficiency_curve(self, discharge_efficiency_curve: dict):
        self.__discharge_efficiency_curve = discharge_efficiency_curve

    @connected_ev.setter
    def connected_ev(self, ev: EV):
        self.__connected_ev = ev

    @incoming_ev.setter
    def incoming_ev(self, ev: EV):
        self.__incoming_ev = ev

    def plug_car(self, car: EV):
        """
        Connects a car to the charger.

        Parameters
        ----------
        car : object
            Car instance to be connected to the charger.

        Raises
        ------
        ValueError
            If the charger has reached its maximum connected cars' capacity.
        """
        #if self.connected_ev is None
        self.connected_ev = car
        print("A conectar car")
        #else:
        #    raise ValueError("Charger has reached its maximum connected cars capacity")

    def unplug_car(self):
        """
        Disconnects a car from the charger.

        Parameters
        ----------
        car : object
            Car instance to be disconnected from the charger.
        """
        self.connected_ev = None
        print("APAGUEI")

    def associate_incoming_car(self, car: EV):
        """
        Associates incoming car to the charger.

        Parameters
        ----------
        car : object
            Car instance to be connected to the charger.

        Raises
        ------
        ValueError
            If the charger has reached its maximum associated cars' capacity.
        """
        #if self.incoming_ev_ev is None:
        self.incoming_ev = car
        print("A cincoming car")

        #else:
        #    raise ValueError("Charger has reached its maximum associated cars capacity")

    def disassociate_incoming_car(self):
        """
        Disassociates incoming car from the charger.

        Parameters
        ----------
        car : object
            Car instance to be disconnected from the charger.
        """
        self.incoming_ev = None
        print("APAGUEI")


    def update_evs_soc(self, energy: float):
        charging = energy >= 0

        if charging:
            current_power_level = min(max(abs(energy), self.min_charging_power), self.max_charging_power)
        else:
            current_power_level = min(max(abs(energy), self.min_discharging_power), self.max_discharging_power)

        if charging:
            efficiency_curve = self.charge_efficiency_curve
        else:
            efficiency_curve = self.discharge_efficiency_curve

        lower_power_level = max([power for power in efficiency_curve if power <= current_power_level])
        upper_power_level = min([power for power in efficiency_curve if power >= current_power_level])

        if lower_power_level == upper_power_level:
            charge_discharge_efficiency = efficiency_curve[lower_power_level]
        else:
            lower_efficiency = efficiency_curve[lower_power_level]
            upper_efficiency = efficiency_curve[upper_power_level]
            charge_discharge_efficiency = lower_efficiency + (current_power_level - lower_power_level) * (
                    upper_efficiency - lower_efficiency) / (upper_power_level - lower_power_level)

        car = self.connected_ev
        energy_kwh = current_power_level * charge_discharge_efficiency * (
                15 / 60)  # Convert the power to energy by multiplying by the time step (15 minutes)

        # Here we call the car's battery's charge method directly, passing the energy (positive for charging, negative for discharging)
        car.battery.charge(energy_kwh if charging else -energy_kwh)

    def next_time_step(self):
        r"""Advance to next `time_step` and set `electricity_consumption` at new `time_step` to 0.0."""

        self.disassociate_incoming_car()
        self.unplug_car()
        self.__electricity_consumption.append(0.0)

    def reset(self):
        """
        Resets the Charger to its initial state by disconnecting all cars.
        """
        self.connected_ev = None
        self.incoming_ev = None
        self.__electricity_consumption = [0.0]

    def autosize(self):  # TODO values
        r"""Autosize charger for an EV.
        """
        self.nominal_power = 7.2
        self.efficiency = 0.95
        self.max_charging_power = 7.2,
        self.min_charging_power = 1.4,
        self.max_discharging_power = 7.2,
        self.min_discharging_power = 0.0,
        self.charge_efficiency_curve = {3.6: 0.95, 7.2: 0.97, 22: 0.98, 50: 0.98}
        self.discharge_efficiency_curve = {3.6: 0.95, 7.2: 0.97, 22: 0.98, 50: 0.98}

    #def __str__(self):
    #    return (
    #        f"Charger ID: {self.charger_id}\n"
    #        f"Max Charging Power: {self.max_charging_power} kW\n"
    #        f"Min Charging Power: {self.min_charging_power} kW\n"
    #        f"Max Discharging Power: {self.max_discharging_power} kW\n"
    #        f"Min Discharging Power: {self.min_discharging_power} kW\n"
    #        f"Charge Efficiency Curve: {self.charge_efficiency_curve}\n"
    #        f"Discharge Efficiency Curve: {self.discharge_efficiency_curve}\n"
    #        f"Currently Connected Cars: {len(self.connected_cars)}\n"
    #        f"Connected Cars IDs: {[car.id for car in self.connected_cars]}"
    #    )
#