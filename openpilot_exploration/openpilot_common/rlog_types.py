"""
Here are some types in the log files generated by the Comma3x.
These types are explicitly used to extract relevant fields from the log files.
The log file types seem to change quite frequently, these types worked 24-04-2024.
Some of the fields may be commented out, this is because they are not used. 
By omitting them in these types they are not extracted from the log files, and do not crash if they are
not found.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Position:
    x: List[float]  # length 33
    y: List[float]  # length 33
    z: List[float]  # length 33
    t: List[float]  # length 33
    xStd: List[float]  # length 33
    yStd: List[float]  # length 33
    zStd: List[float]  # length 33


@dataclass
class XYZT:
    x: List[float]  # length 33
    y: List[float]  # length 33
    z: List[float]  # length 33
    t: List[float]  # length 33


@dataclass
class DisengagePredictions:
    t: List[float]  # length 5
    brakeDisengageProbs: List[float]  # length 5
    gasDisengageProbs: List[float]  # length 5
    steerOverrideProbs: List[float]  # length 5
    brake3MetersPerSecondSquaredProbs: List[float]  # length 5
    brake4MetersPerSecondSquaredProbs: List[float]  # length 5
    brake5MetersPerSecondSquaredProbs: List[float]  # length 5


@dataclass
class MetaData:
    engagedProb: float
    desirePrediction: List[float]  # length 32
    desireState: List[float]  # length 8
    disengagePredictions: DisengagePredictions
    hardBrakePredicted: bool
    # laneChangeState: str
    # laneChangeDirection: str


@dataclass
class LeadsV3:
    prob: float
    probTime: float
    t: List[float]  # length 6
    x: List[float]  # length 6
    y: List[float]  # length 6
    v: List[float]  # length 6
    a: List[float]  # length 6
    xStd: List[float]  # length 6
    yStd: List[float]  # length 6
    vStd: List[float]  # length 6
    aStd: List[float]  # length 6


@dataclass
class TemporalPose:
    trans: List[float]  # length 3
    rot: List[float]  # length 3
    transStd: List[float]  # length 3
    rotStd: List[float]  # length 3


@dataclass
class Action:
    desiredCurvature: float


@dataclass
class ModelV2OutputData:
    position: Position
    orientation: XYZT
    velocity: XYZT
    orientationRate: XYZT
    laneLines: List[XYZT]  # length 4
    laneLineProbs: List[float]
    roadEdges: List[XYZT]  # length 2
    roadEdgeStds: List[float]  # length 2
    meta: MetaData
    laneLineStds: List[float]  # length 4
    # modelExecutionTime: float
    # gpuExecutionTime: float
    leadsV3: List[LeadsV3]  # length 3
    acceleration: XYZT
    # frameIdExtra: int
    temporalPose: TemporalPose
    # navEnabled: bool
    confidence: str
    # locationMonoTime: int
    action: Action


@dataclass
class CameraOdometryOutputData:
    trans: List[float]
    rot: List[float]
    transStd: List[float]
    rotStd: List[float]
    frameId: int
    timestampEof: int
    wideFromDeviceEuler: List[float]
    wideFromDeviceEulerStd: List[float]
    roadTransformTrans: List[float]
    roadTransformTransStd: List[float]


@dataclass
class CarState:
    # @dataclass
    # class CruiseState:
    #     enabled: bool
    #     speed: float
    #     available: bool
    #     speedOffset: float
    #     standstill: bool
    #     nonAdaptive: bool
    #     speedCluster: float

    # @dataclass
    # class Event:
    #     # examples of event names:
    #     # doorOpen, seatbeltNotLatched, wrongGear, wrongCarMode, parkBrake, pcmDisable
    #     name: str
    #     enable: bool
    #     noEntry: bool
    #     warning: bool
    #     userDisable: bool
    #     softDisable: bool
    #     immediateDisable: bool
    #     preEnable: bool
    #     permanent: bool
    #     overrideLongitudinal: bool
    #     overrideLateral: bool

    vEgo: float
    gas: float
    gasPressed: bool
    brake: float
    brakePressed: bool
    steeringAngleDeg: float
    steeringTorque: float
    steeringPressed: bool
    # cruiseState: CruiseState
    # events: List[Event]
    gearShifter: str
    steeringRateDeg: float
    aEgo: float
    vEgoRaw: float
    standstill: bool
    brakeLightsDEPRECATED: bool
    leftBlinker: bool
    rightBlinker: bool
    yawRate: float
    genericToggle: bool
    doorOpen: bool
    seatbeltUnlatched: bool
    canValid: bool
    steeringTorqueEps: float
    clutchPressed: bool
    # steeringRateLimitedDEPRECATED: bool
    # stockAeb: bool
    # stockFcw: bool
    # espDisabled: bool
    leftBlindspot: bool
    rightBlindspot: bool
    steerFaultTemporary: bool
    steerFaultPermanent: bool
    steeringAngleOffsetDeg: float
    brakeHoldActive: bool
    # parkingBrake: bool
    # canTimeout: bool
    # fuelGauge: float
    # accFaulted: bool
    # charging: bool
    # vEgoCluster: float
    # regenBraking: bool
    # engineRpm: float
    # carFaultedNonCritical: bool


@dataclass
# liveCalibration
class LiveCalibration:

    # calStatusDEPRECATED: int
    # calCycle: int
    # calPerc: int
    rpyCalib: List[float]
    # rpyCalibSpread: List[float]
    # validBlocks: int
    # wideFromDeviceEuler: List[float]
    # calStatus: str
    # height: List[float]


@dataclass
class DeviceState:

    # @dataclass
    # class NetworkInfo:
    #     technology: str
    #     operator: str
    #     band: str
    #     channel: int
    #     extra: str
    #     state: str

    # @dataclass
    # class NetworkStats:
    #     wwanTx: int
    #     wwanRx: int

    # cpu0DEPRECATED: int
    # cpu1DEPRECATED: int
    # cpu2DEPRECATED: int
    # cpu3DEPRECATED: int
    # memDEPRECATED: int
    # gpuDEPRECATED: int
    # batDEPRECATED: int
    # freeSpacePercent: float
    # batteryPercentDEPRECATED: int
    # fanSpeedPercentDesired: int
    # started: bool
    # usbOnlineDEPRECATED: bool
    # startedMonoTime: int
    # thermalStatus: str
    # batteryCurrentDEPRECATED: int
    # batteryVoltageDEPRECATED: int
    # chargingErrorDEPRECATED: bool
    # chargingDisabledDEPRECATED: bool
    # memoryUsagePercent: int
    # cpuUsagePercentDEPRECATED: int
    # pa0DEPRECATED: int
    # networkType: str
    # offroadPowerUsageUwh: int
    # networkStrength: str
    # carBatteryCapacityUwh: int
    # cpuTempC: List[float]
    # gpuTempC: List[float]
    # memoryTempC: float
    # batteryTempCDEPRECATED: float
    # ambientTempCDEPRECATED: float
    # networkInfo: NetworkInfo
    # lastAthenaPingTime: int
    # gpuUsagePercent: int
    # cpuUsagePercent: List[int]
    # nvmeTempC: List[float]
    # modemTempC: List[float]
    # screenBrightnessPercent: int
    # pmicTempC: List[float]
    # powerDrawW: float
    # networkMetered: bool
    # somPowerDrawW: float
    # networkStats: NetworkStats
    # maxTempC: float
    deviceType: str


@dataclass
class RoadCameraState:

    frameId: int
    # encodeId: int
    # timestampEof: int
    # frameLengthDEPRECATED: int
    # integLines: int
    # globalGainDEPRECATED: int
    # frameType: str
    # timestampSof: int
    # lensPosDEPRECATED: int
    # lensSagDEPRECATED: float
    # lensErrDEPRECATED: float
    # lensTruePosDEPRECATED: float
    # gain: float
    # recoverStateDEPRECATED: int
    # highConversionGain: bool
    # measuredGreyFraction: float
    # targetGreyFraction: float
    # processingTime: float
    # frameIdSensor: int
    sensor: str
    # exposureValPercent: float
    # requestId: int
