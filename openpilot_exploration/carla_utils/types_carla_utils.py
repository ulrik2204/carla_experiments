from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import (
    Any,
    Callable,
    Dict,
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
    Union,
)

import carla
import numpy as np
from PIL import Image


class Constant:
    def __new__(cls):
        raise TypeError("Cannot instantiate constant class")


TSensorData = TypeVar("TSensorData")

# It is suppsed to have to be a Mapping[str, carla.Actor], but it does not support extensions of carla.Actor


@dataclass
class SensorBlueprint(Generic[TSensorData]):
    """A dataclass representing the name of a sensor blueprint
    as well as its corresponding type of sensor data.

    THIS CLASS SHOULD BE BE INSTANTIATED, USE THE SensorBlueprints CONSTANT CLASS INSTEAD.
    """

    name: str
    sensor_data_type: Type[TSensorData]

    def __hash__(self) -> int:
        return self.name.__hash__()


TActorMap = TypeVar("TActorMap", bound=Mapping[str, Any])
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

SaveItemValue = Union[Dict[str, "SaveItemValue"], np.ndarray, carla.Image, Image.Image]
SaveItems = Dict[str, SaveItemValue]


# TContextContra = TypeVar("TContextContra", bound=BatchContext, contravariant=True)
# TSensorDataMapContra = TypeVar(
#     "TSensorDataMapContra", bound=Mapping[str, Any], contravariant=True
# )


CarlaTask = Union[
    Callable[[TContext], Union[SaveItems, None]],
    Callable[[TContext, TSensorDataMap], Union[SaveItems, None]],
]

# TSaveFileBasePath = TypeVar("TSaveFileBasePath", bound=Optional[Path])

# TContextContra = TypeVar("TContextContra", bound=BatchContext, contravariant=True)

# TSettings = TypeVar("TSettings")


FlexiblePath = Union[str, Path]


# class SegmentResultOptions(TypedDict, Generic[TContext], total=False):
#     on_finish_save_files: Callable[[TContext], SaveItems]
#     on_segment_end: Callable[[TContext, Path], None]
#     cleanup_actors: bool


# class SegmentResult(TypedDict, Generic[TContext, TSensorDataMap]):
#     tasks: List[CarlaTask[TContext, TSensorDataMap]]
#     options: SegmentResultOptions[TContext]


# class Segment(Protocol, Generic[TContext, TSensorDataMap]):
#     def __call__(
#         self, context: TContext
#     ) -> SegmentResult[TContext, TSensorDataMap]: ...


# class DecoratedSegment(Protocol, Generic[TContextContra]):
#     def __call__(self, context: TContextContra, batch_base_path: Path) -> None: ...


# The NEW API


class SegmentResultOptions(TypedDict, Generic[TContext], total=False):
    on_segment_end: Callable[[TContext, Path], None]
    cleanup_actors: bool


class SegmentConfigResult(TypedDict, Generic[TContext, TSensorDataMap]):
    context: TContext
    tasks: List[CarlaTask[TContext, TSensorDataMap]]
    options: SegmentResultOptions[TContext]


TSettingsContra = TypeVar("TSettingsContra", contravariant=True)


class SegmentConfig(Protocol, Generic[TSettingsContra, TContext, TSensorDataMap]):
    def __call__(
        self, settings: TSettingsContra
    ) -> SegmentConfigResult[TContext, TSensorDataMap]: ...


class Segment(Protocol, Generic[TSettingsContra]):
    def __call__(self, settings: TSettingsContra) -> None: ...


# END OF NEW API


# class BatchResultOptions(TypedDict, Generic[TContext], total=False):
#     on_batch_end: Callable[[TContext], None]
#     cleanup_actors: bool


# class BatchResult(TypedDict, Generic[TContext]):
#     context: TContext
#     segments: List[DecoratedSegment[TContext]]
#     options: BatchResultOptions[TContext]


# class Batch(Protocol, Generic[TSettingsContra]):
#     def __call__(self, settings: TSettingsContra) -> BatchResult: ...


# class DecoratedBatch(Protocol, Generic[TSettingsContra]):
#     def __call__(
#         self,
#         base_path: Path,
#         settings: TSettingsContra,
#         on_segment_end: Optional[Callable[[], None]] = None,
#     ) -> None: ...
