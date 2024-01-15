from abc import ABC
from dataclasses import dataclass
from queue import Queue
from typing import Any, Generic, Literal, Mapping, Tuple, Type, TypedDict, TypeVar

import carla


class Constant:
    def __new__(cls):
        raise TypeError("Cannot instantiate constant class")


TSensorData = TypeVar("TSensorData")

# It is suppsed to have to be a Mapping[str, carla.Actor], but it does not support extensions of carla.Actor
TActors = TypeVar("TActors", bound=Mapping[str, Any])


@dataclass
class SensorBlueprint(Generic[TSensorData]):
    name: str
    sensor_data_type: Type[TSensorData]

    def __hash__(self) -> int:
        return self.name.__hash__()


TActorMap = TypeVar("TActorMap")
TSensorsMap = TypeVar("TSensorsMap", bound=Mapping[str, Any])


class SensorConfig(TypedDict, Generic[TSensorData]):
    blueprint: SensorBlueprint[TSensorData]
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    attributes: Mapping[str, str]


AvailableMaps = Literal[
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10",
    "Town11",
    "Town12",
]


@dataclass
class CarlaContext(ABC, Generic[TSensorsMap, TActorMap]):
    """A keyword only dataclass that should be inherited from"""

    client: carla.Client
    # map: carla.Map  Do I need this to enable Opendrive navigation?
    ego_vehicle: carla.Vehicle
    sensor_map: Mapping[str, carla.Sensor]
    sensor_data_queue: Queue
    actor_map: Mapping[str, carla.Actor]
