import random
import time
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional, Tuple, TypedDict

import carla
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from openpilot_exploration.carla_eval.pid_controller import VehiclePIDController
from openpilot_exploration.carla_utils.constants import SensorBlueprints
from openpilot_exploration.carla_utils.setup import (
    BatchContext,
    create_segment,
    setup_carla_client,
)
from openpilot_exploration.carla_utils.spawn import (
    spawn_ego_vehicle,
    spawn_sensors,
    spawn_vehicle_bots,
    spawn_walker_bots,
)
from openpilot_exploration.carla_utils.types_carla_utils import SegmentConfigResult
from openpilot_exploration.common.utils_op_deepdive import (
    T_ANCHORS,
    frd_waypoints_to_fru,
    setup_calling_op_deepdive,
    transform_images,
)
from openpilot_exploration.models.op_deepdive import SequenceBaselineV1

CKPT_PATH = "./.weights/may09_epoch_99.pth"
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


_planning_v0 = load_line_following_model()
# wheeled_speed_pid = PIDController(110, k_i=11.5, rate=int(1 / DT_CTRL))
call_op_deepdive = setup_calling_op_deepdive(_planning_v0, 1, "cuda")


# def _old_predict_vehicle_controls(image):
#     with torch.no_grad():
#         # this now returns waypoints
#         output = planning_v0(image)
#         output = output.squeeze(0)
#         steer = float(output[0].item())
#         throttle = float(output[1].item())
#         brake = float(output[2].item())
#         return steer, throttle, 0.0 if brake < 0.5 else brake


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
    pred_trajectory, pred_conf, *_ = call_op_deepdive(input_image)
    most_confident_trajectory = pred_trajectory[pred_conf.argmax(), :]
    # should be (33, 3)

    return most_confident_trajectory  # Should be [num_pts, 3] where num_pts = 33


def calculate_speeds_tensor(
    waypoints: torch.Tensor, t_anchors: torch.Tensor
) -> np.ndarray:
    # Expects tensor of shape (N, 3) for waypoints and shape (N,) for t_anchors
    # Calculate differences in x, y, and z between successive waypoints
    diffs = waypoints[1:] - waypoints[:-1]
    # Calculate Euclidean distances between successive waypoints
    distances = torch.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2 + diffs[:, 2] ** 2)
    # Get time differences from t_anchors
    time_diffs = t_anchors[1:] - t_anchors[:-1]
    # Avoid division by zero for time differences by replacing 0 with a very small number
    time_diffs = torch.where(time_diffs == 0, torch.tensor(1e-6), time_diffs)
    # Calculate speeds
    speeds = (distances / time_diffs).abs()
    return speeds.cpu().numpy()


def transform_from_ego_frame_to_map_coordinates(
    ego_trans: carla.Transform, waypoint: np.ndarray
) -> carla.Location:
    # IS THIS CORRECT? SHOULDN'T IT JUST ADD THE WAYPOINT TO THE EGO LOCATION?
    fu_c = ego_trans.get_forward_vector().make_unit_vector()
    forward_unit_vec = np.array((fu_c.x, fu_c.y, fu_c.z))
    ru_c = ego_trans.get_right_vector().make_unit_vector()
    right_unit_vec = np.array((ru_c.x, ru_c.y, ru_c.z))
    up = ego_trans.get_up_vector().make_unit_vector()
    up_unit_vec = np.array((up.x, up.y, up.z))

    ego_l = np.array((ego_trans.location.x, ego_trans.location.y, ego_trans.location.z))
    # This neglects the z coordinate
    target_l = (
        ego_l
        + waypoint[0] * forward_unit_vec
        + waypoint[1] * right_unit_vec
        + waypoint[2] * up_unit_vec
    )
    # print("trans loc and rot", ego_l)
    # print("trans_matrix", ego_trans.get_matrix())
    return carla.Location(x=target_l[0], y=target_l[1], z=target_l[2])


def draw_trajectory(world: carla.World, points: List[carla.Location]):
    for i in range(len(points) - 1):
        world.debug.draw_line(
            points[i],
            points[i + 1],
            thickness=0.1,
            color=carla.Color(255, 0, 0),
            life_time=0.1,
        )


did_once = False


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
    current_transform = ego_vehicle.get_transform()
    last_image = context.data_dict["last_image"]
    if last_image is None:
        context.data_dict["last_image"] = current_image
        return
    wp_index = 3  # Is way longer into the future than 1, which is good for PID
    trajectory = get_predicted_trajectory(current_image, last_image).cpu()
    # METHOD 1: Converting ECEF waypoints back to ECEF coords and then to CARLA coords
    # vehicle_location_ecef = carla_location_to_ecef(
    #     context.map, current_transform.location
    # )
    # vehicle_rotation_ecef = carla_rotation_to_ecef_frd_quaternion(
    #     context.map, current_transform.rotation
    # )
    # ecef_location, ecef_rotation = waypoints_to_ecef(
    #     trajectory.numpy(), vehicle_location_ecef, vehicle_rotation_ecef
    # )
    # ecef_rotation_target = ecef_rotation[wp_index]
    # target_rotation = ecef_frd_quaternion_to_carla_rotation(
    #     context.map, ecef_rotation_target
    # )
    # trajecotry_locations = [
    #     ecef_to_carla_location(context.map, loc) for loc in ecef_location
    # ]
    # target_location = trajecotry_locations[wp_index]
    # METHOD 2: The waypoints are in FRD, convert to left-handed FRU and then convert to CARLA
    fru_waypoints = frd_waypoints_to_fru(trajectory.numpy())
    # fru_waypoints[:, 1] = -fru_waypoints[:, 1]  # pretty sure this is wrong
    # print("fru_waypoint", fru_waypoints[wp_index])
    trajectory_locations = [
        transform_from_ego_frame_to_map_coordinates(current_transform, wp)
        for wp in fru_waypoints
    ]
    used_wp = fru_waypoints[wp_index]
    # print("average_wp", average_wp)

    target_location = transform_from_ego_frame_to_map_coordinates(
        current_transform, used_wp
    )

    draw_trajectory(context.client.get_world(), trajectory_locations)
    # print("converted_waypoint", converted_waypoint)
    print(f"{trajectory.shape}, {T_ANCHORS.shape}")
    speeds = calculate_speeds_tensor(trajectory, torch.tensor(T_ANCHORS))
    next_speed = speeds[wp_index]
    next_speed = np.linalg.norm(
        (used_wp[:2] - fru_waypoints[:2, 0]) / (T_ANCHORS[wp_index] - T_ANCHORS[0])
    ).item()
    print("next_speed", next_speed)
    # print("next_speed", next_speedo)
    # loc = current_transform.location
    # print("-----")
    # print("current location", loc.x, loc.y, loc.z)
    # print("target location", target_location.x, target_location.y, target_location.z)
    control = context.pid_controller.run_step(next_speed, target_location)
    # control.brake = 0
    # print("control", control)
    ego_vehicle.apply_control(control)
    context.data_dict["last_image"] = current_image


done = False


def spectator_follow_ego_vehicle_task(
    context: AppContext, sensor_data_map: AppSensorDataMap
) -> None:
    global done
    # if done:
    #     return
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    ego_loc = vehicle_transform.location
    forward = vehicle_transform.get_forward_vector()
    spectator = context.client.get_world().get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = carla.Location(
        x=ego_loc.x - 10 * forward.x,
        y=ego_loc.y - 10 * forward.y,
        z=ego_loc.z + 10,
    )
    rot = vehicle_transform.rotation
    spectator_transform.rotation = carla.Rotation(pitch=-30, yaw=rot.yaw, roll=0)
    spectator.set_transform(spectator_transform)
    # done = True


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
    def infinite_segment_config(settings: AppSettings) -> SegmentConfigResult:

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
        spawn_point_index = 5
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
        sensor_map = spawn_sensors(
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
        pid_controller = VehiclePIDController(
            ego_vehicle,
            {"K_P": 1.0, "K_D": 0.0, "K_I": 0.0, "dt": 1 / 20},
            {"K_P": 1.0, "K_D": 0.0, "K_I": 0.0, "dt": 1 / 20},
        )

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
    seg = generate_infinite_segment("Town04")
    # chunks = {"Chunk_1": [batch1]}
    carla_client = setup_carla_client()
    settings = AppSettings(frame_rate=20, client=carla_client)
    seg(settings)


if __name__ == "__main__":
    main()
