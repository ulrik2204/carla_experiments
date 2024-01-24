from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from queue import Queue
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
)

import carla


class Constant:
    def __new__(cls):
        raise TypeError("Cannot instantiate constant class")


TSensorData = TypeVar("TSensorData")

# It is suppsed to have to be a Mapping[str, carla.Actor], but it does not support extensions of carla.Actor


@dataclass
class SensorBlueprint(Generic[TSensorData]):
    name: str
    sensor_data_type: Type[TSensorData]

    def __hash__(self) -> int:
        return self.name.__hash__()


TActorMap = TypeVar("TActorMap")
TSensorMap = TypeVar("TSensorMap")


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


@dataclass  # kw_only=True
class BatchContext(ABC, Generic[TSensorMap, TActorMap]):
    """A keyword only dataclass that should be inherited from"""

    client: carla.Client
    map: carla.Map
    ego_vehicle: carla.Vehicle
    sensor_map: TSensorMap
    sensor_data_queue: Queue
    actor_map: TActorMap


TContext = TypeVar("TContext", bound=BatchContext)
TSensorDataMap = TypeVar("TSensorDataMap", bound=Mapping[str, Any])
TActorMap = TypeVar("TActorMap", bound=Mapping[str, Any])


class SegmentResult(TypedDict, Generic[TContext, TSensorDataMap]):
    tasks: List[Callable[[TContext, TSensorDataMap], None]]
    on_exit: Optional[Callable[[TContext], None]]


class Segment(Protocol, Generic[TContext, TSensorDataMap]):
    def __call__(self, context: TContext) -> SegmentResult[TContext, TSensorDataMap]:
        ...


TContextContra = TypeVar("TContextContra", bound=BatchContext, contravariant=True)

TSettings = TypeVar("TSettings")


class DecoratedSegment(Protocol, Generic[TContextContra]):
    def __call__(self, context: TContextContra) -> None:
        ...


class BatchResult(TypedDict, Generic[TContext]):
    context: TContext
    segments: List[DecoratedSegment[TContext]]
    on_exit: Optional[Callable[[TContext], None]]


TSettingsContra = TypeVar("TSettingsContra", contravariant=True)


class Batch(Protocol, Generic[TSettingsContra]):
    def __call__(self, settings: TSettingsContra) -> BatchResult:
        ...


class DecoratedBatch(Protocol, Generic[TSettingsContra]):
    def __call__(self, settings: TSettingsContra) -> None:
        ...
