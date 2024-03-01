import random
import time
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional, Tuple, TypedDict

import carla
import cv2
import numpy as np
import pymap3d as pm
import torch
import torchvision.transforms as transforms
from matplotlib import axis
from PIL import Image

from carla_experiments.carla_eval.pid_controller import VehiclePIDController
from carla_experiments.carla_utils.constants import SensorBlueprints
from carla_experiments.carla_utils.setup import (
    BatchContext,
    create_segment,
    setup_carla_client,
    setup_sensors,
)
from carla_experiments.carla_utils.spawn import (
    spawn_ego_vehicle,
    spawn_vehicle_bots,
    spawn_walker_bots,
)
from carla_experiments.carla_utils.types_carla_utils import FullSegmentConfigResult
from carla_experiments.models.op_deepdive import SequenceBaselineV1

CKPT_PATH = "./.weights/opd_carla_epoch_98.pth"
DT_CTRL = 0.01  # controlsd
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()


def load_line_following_model():
    model = SequenceBaselineV1(5, 33, 1.0, 0.0, "adamw")
    model.load_state_dict(torch.load(CKPT_PATH))
    model.to(DEVICE)
    # planning_v0.eval().cuda()
    # model, *_ = load_state_dict(model, None, "./.weights/agentB0/091148-loss0.0007.pt")
    return model


planning_v0 = load_line_following_model()
# wheeled_speed_pid = PIDController(110, k_i=11.5, rate=int(1 / DT_CTRL))


def _old_predict_vehicle_controls(image):
    with torch.no_grad():
        # this now returns waypoints
        output = planning_v0(image)
        output = output.squeeze(0)
        steer = float(output[0].item())
        throttle = float(output[1].item())
        brake = float(output[2].item())
        return steer, throttle, 0.0 if brake < 0.5 else brake


class AppActorMap(TypedDict):
    vehicles: List[carla.Vehicle]
    walkers: List[Tuple[carla.WalkerAIController, carla.Walker]]


class AppSensorMap(TypedDict):
    front_camera: carla.Sensor


class AppSensorDataMap(TypedDict):
    front_camera: carla.Image


@dataclass
class AppSettings:
    frame_rate: int
    client: carla.Client


class DataDict(TypedDict):
    last_location: Optional[np.ndarray]
    last_rotation: Optional[np.ndarray]
    last_image: Optional[Image.Image]


@dataclass
class AppContext(BatchContext[AppSensorMap, AppActorMap], AppSettings):
    data_dict: DataDict
    pid_controller: VehiclePIDController


# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6

medmodel_fl = 910.0
medmodel_intrinsics = np.array(
    [
        [medmodel_fl, 0.0, 0.5 * MEDMODEL_INPUT_SIZE[0]],
        [0.0, medmodel_fl, MEDMODEL_CY],
        [0.0, 0.0, 1.0],
    ]
)
DEVICE_FRAME_FROM_VIEW_FRAME = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)
VIEW_FRAME_FROM_DEVICE_FRAME = DEVICE_FRAME_FROM_VIEW_FRAME.T
NUM_PTS = 10 * 20  # 10 s * 20 Hz = 200 frames
T_ANCHORS = np.array(
    (
        0.0,
        0.00976562,
        0.0390625,
        0.08789062,
        0.15625,
        0.24414062,
        0.3515625,
        0.47851562,
        0.625,
        0.79101562,
        0.9765625,
        1.18164062,
        1.40625,
        1.65039062,
        1.9140625,
        2.19726562,
        2.5,
        2.82226562,
        3.1640625,
        3.52539062,
        3.90625,
        4.30664062,
        4.7265625,
        5.16601562,
        5.625,
        6.10351562,
        6.6015625,
        7.11914062,
        7.65625,
        8.21289062,
        8.7890625,
        9.38476562,
        10.0,
    )
)
T_IDX = np.linspace(0, 10, num=NUM_PTS)


def quat2rot(quats):
    quats = np.array(quats)
    input_shape = quats.shape
    quats = np.atleast_2d(quats)
    Rs = np.zeros((quats.shape[0], 3, 3))
    q0 = quats[:, 0]
    q1 = quats[:, 1]
    q2 = quats[:, 2]
    q3 = quats[:, 3]
    Rs[:, 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    Rs[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    Rs[:, 0, 2] = 2 * (q0 * q2 + q1 * q3)
    Rs[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    Rs[:, 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    Rs[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    Rs[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    Rs[:, 2, 1] = 2 * (q0 * q1 + q2 * q3)
    Rs[:, 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    if len(input_shape) < 2:
        return Rs[0]
    else:
        return Rs


def calibration(extrinsic_matrix, cam_intrinsics, device_frame_from_road_frame=None):
    if device_frame_from_road_frame is None:
        device_frame_from_road_frame = np.hstack(
            (np.diag([1, -1, -1]), [[0], [0], [1.51]])
        )
    med_frame_from_ground = (
        medmodel_intrinsics
        @ VIEW_FRAME_FROM_DEVICE_FRAME
        @ device_frame_from_road_frame[:, (0, 1, 3)]
    )
    ground_from_med_frame = np.linalg.inv(med_frame_from_ground)

    extrinsic_matrix_eigen = extrinsic_matrix[:3]
    camera_frame_from_road_frame = np.dot(cam_intrinsics, extrinsic_matrix_eigen)
    camera_frame_from_ground = np.zeros((3, 3))
    camera_frame_from_ground[:, 0] = camera_frame_from_road_frame[:, 0]
    camera_frame_from_ground[:, 1] = camera_frame_from_road_frame[:, 1]
    camera_frame_from_ground[:, 2] = camera_frame_from_road_frame[:, 3]
    warp_matrix = np.dot(camera_frame_from_ground, ground_from_med_frame)

    return warp_matrix


def transform_images(current_image: Image.Image, last_image: Image.Image):
    # seq_input_img
    trans = transforms.Compose(
        [
            # transforms.Resize((900 // 2, 1600 // 2)),
            # transforms.Resize((9 * 32, 16 * 32)),
            transforms.Resize((128, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.3890, 0.3937, 0.3851], [0.2172, 0.2141, 0.2209]),
        ]
    )
    warp_matrix = calibration(
        extrinsic_matrix=np.array(
            [[0, -1, 0, 0], [0, 0, -1, 1.22], [1, 0, 0, 0], [0, 0, 0, 1]]
        ),
        cam_intrinsics=np.array([[910, 0, 582], [0, 910, 437], [0, 0, 1]]),
        device_frame_from_road_frame=np.hstack(
            (np.diag([1, -1, -1]), [[0], [0], [1.22]])
        ),
    )
    imgs = [current_image, last_image]  # contains one more img
    imgs = [
        cv2.warpPerspective(
            src=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
            M=warp_matrix,
            dsize=(512, 256),
            flags=cv2.WARP_INVERSE_MAP,
        )
        for img in imgs
    ]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = list(Image.fromarray(img) for img in imgs)
    imgs = list(trans(img)[None] for img in imgs)
    input_img = torch.cat(imgs, dim=0)  # [N+1, 3, H, W]
    del imgs
    input_img = torch.cat((input_img[:-1, ...], input_img[1:, ...]), dim=1)
    return input_img.to(DEVICE)  # Should be [1, 6, 128, 256]


# def transform_pose(current_location, last_location, current_rotation, last_rotation):
#     frame_locations = np.array([current_location, last_location])
#     frame_orientations = np.array([current_rotation, last_rotation])
#     future_poses = []
#     # fix_seq_length = 800 in train and 999 in val
#     ecef_from_local = quat2rot(current_rotation)
#     local_from_ecef = ecef_from_local.T
#     frame_positions_local = np.einsum(
#         "ij,kj->ki", local_from_ecef, frame_locations - current_location
#     ).astype(np.float32)

#     # Time-Anchor like OpenPilot
#     fs = [interp1d(T_IDX, frame_positions_local[i : i + NUM_PTS, j]) for j in range(3)]
#     interp_positions = [fs[j](T_ANCHORS)[:, None] for j in range(3)]
#     interp_positions = np.concatenate(interp_positions, axis=1)

#     future_poses.append(interp_positions)
#     future_poses = torch.tensor(np.array(future_poses), dtype=torch.float32)

#     # Really? Predicting 200 points per frame? Are these the 200 next waypoints?

#     return future_poses  # Should be [1, num_pts, 3] where num_pts = 200


def get_predicted_trajectory(
    current_image,
    last_image,
) -> torch.Tensor:
    input_image = transform_images(current_image, last_image)
    bs = 1
    hidden = torch.zeros((2, bs, 512)).to(DEVICE)
    with torch.no_grad():

        pred_cls, pred_trajectory, hidden = planning_v0(input_image, hidden)
    pred_conf = torch.softmax(pred_cls, dim=-1)[0]
    # pred_trajectory.Shape = (5, 33, 3)
    pred_trajectory = pred_trajectory.reshape(planning_v0.M, planning_v0.num_pts, 3)
    # print("pred_trajectory", pred_trajectory.shape)  # (5, 33, 3)
    # The code in training to predict from one batch (data):
    # seq_inputs, seq_labels = (
    #     data["seq_input_img"].cuda(),
    #     data["seq_future_poses"].cuda(),
    # )
    # bs = seq_labels.size(0)
    # seq_length = seq_labels.size(1)

    # hidden = torch.zeros((2, bs, 512)).cuda()
    # total_loss = 0
    # for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm, position=2):
    #     num_steps += 1
    #     inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
    #     pred_cls, pred_trajectory, hidden = model(inputs, hidden)
    # TODO: Convert its reference waypoints to CARLA waypoints
    most_confident_trajectory = pred_trajectory[pred_conf.argmax(), :]
    # should be (33, 3)

    return most_confident_trajectory  # Should be [num_pts, 3] where num_pts = 33


def calculate_speeds_tensor(waypoints: torch.Tensor) -> np.ndarray:
    # Expects tensor of shape (N, 3) where N is the number of waypoints
    # Calculate differences in x, y, and t between successive waypoints
    diffs = waypoints[1:] - waypoints[:-1]
    # Calculate Euclidean distances between successive waypoints (ignoring time dimension)
    distances = torch.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
    # Get time differences
    time_diffs = diffs[:, 2]
    # Avoid division by zero for time differences by replacing 0 with a very small number
    time_diffs = torch.where(time_diffs == 0, torch.tensor(1e-6), time_diffs)
    # Calculate speeds
    speeds = (distances / time_diffs).abs()
    return speeds.cpu().numpy()


def calculate_speeds_from_carla_waypoints(
    waypoints: List[carla.Waypoint],
) -> List[float]:
    # Initialize an empty list to hold the distances
    distances = []

    # Calculate distances between successive waypoints
    for i in range(1, len(waypoints)):
        location1 = waypoints[i - 1].transform.location
        location2 = waypoints[i].transform.location

        # Calculate the Euclidean distance between waypoints
        distance = (
            (location2.x - location1.x) ** 2 + (location2.y - location1.y) ** 2
        ) ** 0.5
        distances.append(distance)

    # Convert distances to a PyTorch tensor
    distances_tensor = np.array(distances)

    # Since waypoints are sampled at 20Hz, the time delta between each is 1/20 seconds or 0.05 seconds
    delta_time = 0.05

    # Calculate speeds by dividing distances by the time delta
    speeds = distances_tensor / delta_time

    return speeds.tolist()


def convert_ecef_waypoint_to_carla_waypoint(carla_map: carla.Map, waypoint: np.ndarray):
    ecef_x = waypoint[0]
    ecef_y = waypoint[1]
    ecef_z = waypoint[2]

    # ECEF to ENU
    origin_geolocation = carla_map.transform_to_geolocation(carla.Location(0, 0, 0))
    latitude = np.radians(origin_geolocation.latitude)
    longitude = np.radians(origin_geolocation.longitude)
    altitude = origin_geolocation.altitude  # In meters

    x, y, z = pm.ecef2enu(
        ecef_x,
        ecef_y,
        ecef_z,
        latitude,
        longitude,
        altitude,
        deg=False,
    )

    # ENU to ESU into Carla waypoint
    location = carla.Location(x, -y, z)

    # TODO: Add orientation as well
    return location


def predict_vehicle_controls_task(
    context: AppContext, sensor_data_map: AppSensorDataMap
):
    front_image = sensor_data_map["front_camera"]
    current_image = carla_image_to_pil_image(front_image)
    # radar_data = parse_radar_data(sensor_data_map["radar"])
    # imu_data = parse_imu_data(sensor_data_map["imu"])
    # gnss_data = parse_gnss_data(sensor_data_map["gnss"])
    # speed = calculate_vehicle_speed(context.ego_vehicle)
    # front_image.timestamp  # TODO: use this for frame times?
    ego_vehicle = context.ego_vehicle
    last_image = context.data_dict["last_image"]
    if last_image is None:
        context.data_dict["last_image"] = current_image
        return
    trajectory = get_predicted_trajectory(current_image, last_image)
    next_waypoint = torch.mean(trajectory[1:5], axis=0).cpu().numpy()  # type: ignore
    print("trajectory", trajectory.shape, next_waypoint, trajectory[6])
    converted_waypoint = convert_ecef_waypoint_to_carla_waypoint(
        context.map, next_waypoint
    )
    print("converted_waypoint", converted_waypoint)
    speeds = calculate_speeds_tensor(trajectory)
    next_speed = np.mean(speeds[1:5], axis=0)
    print("next_speed", next_speed)
    control = context.pid_controller.run_step(next_speed, converted_waypoint)
    # control.brake = 0
    print("control", control)
    ego_vehicle.apply_control(control)
    context.data_dict["last_image"] = current_image
    # print(f"after add [frame {frame}]", len(context.data_dict["images"]))
    # return {
    #     "location": {
    #         str(frame): location_np,
    #     },
    #     "rotation": {
    #         str(frame): rotation_np,
    #     },
    #     "images": {
    #         str(frame): front_image,
    #     },
    # }


def spectator_follow_ego_vehicle_task(
    context: AppContext, sensor_data_map: AppSensorDataMap
) -> None:
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    spectator = context.client.get_world().get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = vehicle_transform.location + carla.Location(z=2)
    spectator_transform.rotation = vehicle_transform.rotation
    spectator.set_transform(spectator_transform)


def configure_traffic_manager(
    traffic_manager: carla.TrafficManager,
    ego_vehicle: carla.Vehicle,
    vehicle_bots: List[carla.Vehicle],
) -> None:
    # traffic_manager.set_random_device_seed(42)
    # traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_respawn_dormant_vehicles(True)
    traffic_manager.set_boundaries_respawn_dormant_vehicles(25, 700)

    # traffic_manager.set_desired_speed(ego_vehicle, 50 / 3.6)  # 50 km/h
    # traffic_manager.set_hybrid_physics_mode(True)
    # traffic_manager.set_hybrid_physics_radius(70)

    for bot in vehicle_bots:
        traffic_manager.ignore_lights_percentage(bot, 5)
        traffic_manager.ignore_signs_percentage(bot, 5)
        traffic_manager.ignore_walkers_percentage(bot, 1)
        traffic_manager.vehicle_percentage_speed_difference(
            bot, random.randint(-30, 30)
        )
        traffic_manager.random_left_lanechange_percentage(bot, random.randint(1, 60))
        traffic_manager.random_right_lanechange_percentage(bot, random.randint(1, 60))
        traffic_manager.update_vehicle_lights(bot, True)


def carla_image_to_pil_image(image: carla.Image) -> Image.Image:
    array = np.frombuffer(np.copy(image.raw_data), dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # to BGRA image
    array = array[:, :, :3][:, :, ::-1]  # Convert to RGB
    return Image.fromarray(array)


def configure_traffic_lights(world: carla.World):
    actors = world.get_actors().filter("traffic.traffic_light")
    for actor in actors:
        if isinstance(actor, carla.TrafficLight):
            actor.set_state(carla.TrafficLightState.Green)
            actor.set_green_time(5)
            actor.set_yellow_time(1)
            actor.set_red_time(5)


def generate_infinite_segment(map: str):
    def infinite_segment_config(settings: AppSettings) -> FullSegmentConfigResult:

        # client = setup_carla_client("Town10HD")
        client = settings.client
        if map not in client.get_world().get_map().name:
            print("Loading new map", map, "...")
            client.load_world(map, reset_settings=False)
            client.reload_world(reset_settings=False)
            time.sleep(5)
        else:
            print("Using previously loaded map", map)
        world = client.get_world()
        world.tick()
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        # len_spawn_points = len(spawn_points)
        spawn_point_index = 1
        # rotation = used_spawn_point.rotation
        # location = used_spawn_point.location
        # print("Generating in map", map)
        # print(
        #     "Using spawn point",
        #     spawn_point_index,
        #     "of ",
        #     len_spawn_points,
        #     "with coords",
        #     f"({location.x:.2f}, {location.y:.2f}, {location.z:.2f})",
        #     "and rotation",
        #     f"({rotation.roll:.2f}, {rotation.pitch:.2f}, {rotation.yaw:.2f})",
        # )
        spawn_point = spawn_points[spawn_point_index]
        ego_vehicle = spawn_ego_vehicle(world, autopilot=False, spawn_point=spawn_point)
        world.set_pedestrians_cross_factor(0.1)
        sensor_data_queue = Queue()
        # TODO: Check sensor positions
        sensor_map = setup_sensors(
            world,
            ego_vehicle,
            sensor_data_queue=sensor_data_queue,
            return_sensor_map_type=AppSensorMap,
            sensor_config={
                "front_camera": {
                    "blueprint": SensorBlueprints.CAMERA_RGB,
                    "location": (2, 0, 1),
                    "rotation": (0, 0, 0),
                    "attributes": {"image_size_x": "1164", "image_size_y": "874"},
                },
            },
        )
        vehicle_bot_spawn_points = spawn_points.copy()
        vehicle_bot_spawn_points.pop(spawn_point_index)
        vehicle_bots = spawn_vehicle_bots(
            world, 10, accessible_spawn_points=vehicle_bot_spawn_points
        )
        walker_bots = spawn_walker_bots(world, 15)
        traffic_manager = client.get_trafficmanager()
        configure_traffic_manager(traffic_manager, ego_vehicle, vehicle_bots)
        configure_traffic_lights(world)

        # client.get_trafficmanager().set_global_distance_to_leading_vehicle()
        data_dict: DataDict = {
            "last_location": None,
            "last_rotation": None,
            "last_image": None,
        }
        pid_controller = VehiclePIDController(ego_vehicle)

        context = AppContext(
            client=client,
            map=carla_map,
            sensor_map=sensor_map,
            sensor_data_queue=sensor_data_queue,
            actor_map={"vehicles": vehicle_bots, "walkers": walker_bots},
            ego_vehicle=ego_vehicle,
            frame_rate=settings.frame_rate,
            data_dict=data_dict,
            pid_controller=pid_controller,
        )
        return {
            "context": context,
            "tasks": [
                spectator_follow_ego_vehicle_task,
                predict_vehicle_controls_task,
            ],
            "options": {
                "cleanup_actors": True,
            },
        }

    return create_segment(None, None, infinite_segment_config)


def main():
    # Comma2k19 called this each batch by the date
    # TODO: Change back to Town01
    seg = generate_infinite_segment("Town05")
    # chunks = {"Chunk_1": [batch1]}
    carla_client = setup_carla_client()
    settings = AppSettings(frame_rate=20, client=carla_client)
    seg(settings)


if __name__ == "__main__":
    main()
