from typing import Optional

import carla
import numpy as np
import pymap3d as pm
from scipy.spatial.transform import Rotation as R


def estimate_rotation_from_direction(
    current_pos, next_pos, up=np.array([0, 0, 1]), epsilon=1e-6
) -> Optional[np.ndarray]:
    forward = next_pos - current_pos
    forward_magnitude = np.linalg.norm(forward)

    # Check for near-identical waypoints
    if forward_magnitude < epsilon:
        # Use the last known rotation if waypoints are too close
        return None
    forward /= forward_magnitude

    right = np.cross(up, forward)
    right_magnitude = np.linalg.norm(right)
    if right_magnitude < epsilon:
        # Fallback strategy here (e.g., use last valid right or a default)
        return None
    right /= np.linalg.norm(right)

    up_corrected = np.cross(forward, right)
    rot_matrix = np.vstack([right, up_corrected, forward])

    rotation = R.from_matrix(rot_matrix).as_quat()
    return np.array([rotation[3], rotation[0], rotation[1], rotation[2]])


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


def waypoints_to_ecef(
    waypoints: np.ndarray,
    initial_frame_position: np.ndarray,
    initial_frame_orientation: np.ndarray,
):
    # Placeholder for the function to estimate rotations based on direction of travel

    # Initialize arrays for positions and rotations
    ecef_positions = [initial_frame_position]
    rotations = [initial_frame_orientation]

    for i in range(len(waypoints) - 1):
        # Estimate rotation based on the direction of travel
        quat_return = estimate_rotation_from_direction(waypoints[i], waypoints[i + 1])
        quat = quat_return if quat_return is not None else rotations[-1]
        rotations.append(quat)

        # Convert the current waypoint to ECEF using the newly estimated rotation
        rot_matrix = quat2rot(quat)
        position_ecef = (
            np.einsum("ij,j->i", rot_matrix, waypoints[i]) + initial_frame_position
        )
        ecef_positions.append(position_ecef)

    # Convert lists to numpy arrays for consistency and easier handling
    ecef_positions = np.array(ecef_positions)
    rotations = np.array(rotations)

    return ecef_positions, rotations


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # Pre-calculate sine and cosine for pitch, yaw, and roll
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Calculate the quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # TODO: May need to change something to convert from left-handed to right-handed coordinate system

    # Need to negate y and z to convert from left-handed to right-handed coordinate system
    return np.array([w, x, y, z])


# HERE: The one currently in use
def carla_rotation_to_ecef_frd_quaternion(
    map: carla.Map, rotation: carla.Rotation
) -> np.ndarray:
    """Takes in a carla.Rotation object which is an euler angle that rotates from the CARLA coordinate system to FRU,
    and converts it to a quaternion that rotates from ECEF to FRD.


    Args:
        map (carla.Map): The map object from the CARLA simulation.
        rotation (carla.Rotation): The rotation in CARLA's coordinate system from CARLA to FRU.

    Returns:
        np.ndarray: A np.ndarray representing the quaternion that rotates from ECEF to FRD on the form [w, x, y, z].
    """
    origin_geolocation = map.transform_to_geolocation(carla.Location(0, 0, 0))
    lat0 = np.radians(origin_geolocation.latitude)
    lon0 = np.radians(origin_geolocation.longitude)
    # alt0 = origin_geolocation.altitude

    # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    # https://carla.readthedocs.io/en/latest/python_api/#carlarotation
    # https://se.mathworks.com/help/uav/ug/coordinate-systems-for-unreal-engine-simulation-in-uav-toolbox.html
    # roll is rotation around the x-axis [-180, 180]
    # pitch is right-handed rotation around the y-axis [-90, 90]
    # yaw is rotation around the z-axis [-180, 180]
    # In CARLA, the rotation of the vehicle and its attached camera are equal
    # In CARLA, the rotation is from the CARLA coordinate frame to FRU (no change).
    # In comma2k19, the rotation is from right-handed ECEF to FRD camera frame.

    # STEP 1: Convert from Unreal rotation ESU -> FRU to ENU rotation ENU -> FRD
    # To account for FRU to FRD: (roll_frd, pitch_frd, yaw_frd) = (roll_enu, pitch_enu + pi, yaw_enu + pi)
    # Convert from CARLA/Unreal to ENU: (roll_enu, pitch_enu, yaw_enu) = (roll_unreal, -pitch_unreal, -yaw_unreal)
    x = np.radians(rotation.roll)
    y = -(np.radians(rotation.pitch) + np.pi)
    z = -(np.radians(rotation.yaw) + np.pi)
    enu_rot = R.from_euler("xyz", (x, y, z), degrees=False)

    # STEP 2: Convert from ENU to ECEF
    r_enu_to_ecef = R.from_matrix(
        [
            [-np.sin(lon0), -np.cos(lon0) * np.sin(lat0), np.cos(lon0) * np.cos(lat0)],
            [np.cos(lon0), -np.sin(lon0) * np.sin(lat0), np.sin(lon0) * np.cos(lat0)],
            [0, np.cos(lat0), np.sin(lat0)],
        ]
    )
    ecef_rot = r_enu_to_ecef * enu_rot

    # STEP 3: Convert to quaternion
    # This returns the quaternion on the [x, y, z, w] format
    ecef_orientation = ecef_rot.as_quat()
    x = ecef_orientation[0]
    y = ecef_orientation[1]
    z = ecef_orientation[2]
    w = ecef_orientation[3]

    return np.array([w, x, y, z])


# NOT USED, BUT WORKS
def carla_location_to_ecef2(map: carla.Map, location: carla.Location):
    origin_geolocation = map.transform_to_geolocation(carla.Location(0, 0, 0))
    lat0 = np.radians(origin_geolocation.latitude)
    lon0 = np.radians(origin_geolocation.longitude)
    alt0 = origin_geolocation.altitude
    x0, y0, z0 = pm.geodetic2ecef(lat0, lon0, alt0, deg=False)
    r = R.from_matrix(
        [
            [-np.sin(lon0), -np.cos(lon0) * np.sin(lat0), np.cos(lon0) * np.cos(lat0)],
            [np.cos(lon0), -np.sin(lon0) * np.sin(lat0), np.sin(lon0) * np.cos(lat0)],
            [0, np.cos(lat0), np.sin(lat0)],
        ]
    )
    enu_pos = np.array([location.x, -location.y, location.z])
    ecef_pos = r.as_matrix() @ enu_pos + (x0, y0, z0)
    return ecef_pos


# HERE: The one currently in use
def carla_location_to_ecef(map: carla.Map, location: carla.Location):
    # I want to check if this finds the origin
    origin_geolocation = map.transform_to_geolocation(carla.Location(0, 0, 0))
    # geolocation = map.transform_to_geolocation(location)
    # https://docs.unrealengine.com/4.27/en-US/BuildingWorlds/Georeferencing/
    e_enu = location.x
    # Negate y to convert from right-handed to left-handed coordinate system
    n_enu = -location.y
    u_enu = location.z

    latitude = np.radians(origin_geolocation.latitude)
    longitude = np.radians(origin_geolocation.longitude)
    altitude = origin_geolocation.altitude  # In meters

    x, y, z = pm.enu2ecef(
        e_enu,
        n_enu,
        u_enu,
        latitude,
        longitude,
        altitude,
        deg=False,
    )
    # latitude = geolocation.latitude
    # longitude = geolocation.longitude
    # altitude = geolocation.altitude

    return np.array([x, y, z])


def ecef_to_carla_location(map: carla.Map, ecef_location: np.ndarray):
    origin_geolocation = map.transform_to_geolocation(carla.Location(0, 0, 0))
    origin_lat = np.radians(origin_geolocation.latitude)
    origin_long = np.radians(origin_geolocation.longitude)
    origin_alt = origin_geolocation.altitude

    x_enu, y_enu, z_enu = pm.ecef2enu(
        ecef_location[0],
        ecef_location[1],
        ecef_location[2],
        origin_lat,
        origin_long,
        origin_alt,
        deg=False,
    )
    x_carla = x_enu
    y_carla = -y_enu
    z_carla = z_enu
    return carla.Location(x_carla, y_carla, z_carla)


def ecef_frd_quaternion_to_carla_rotation(
    carla_map: carla.Map, ecef_rotation: np.ndarray
) -> carla.Rotation:
    """Takes a rotation from ECEF to FRD and converts it to a carla.Rotation from CARLA's coordinate system to FRU.

    Args:
        carla_map (carla.Map): The map object from the CARLA simulation.
        ecef_rotation (np.ndarray): The quaternion representing the rotation from ECEF to FRD on the form [w, x, y, z].

    Returns:
        carla.Rotation: The rotation in CARLA's coordinate system from CARLA to FRU.
    """
    origin_geolocation = carla_map.transform_to_geolocation(carla.Location(0, 0, 0))
    # print("origin_geolocation =
    #   ", origin_geolocation.latitude, origin_geolocation.longitude, origin_geolocation.altitude)
    lat0 = np.radians(origin_geolocation.latitude)
    lon0 = np.radians(origin_geolocation.longitude)

    # STEP 1: Convert from ECEF to ENU
    r_ecef = R.from_quat(
        np.array(
            [ecef_rotation[1], ecef_rotation[2], ecef_rotation[3], ecef_rotation[0]]
        )
    )
    r_ecef_to_enu = R.from_matrix(
        [
            [-np.sin(lon0), -np.cos(lon0) * np.sin(lat0), np.cos(lon0) * np.cos(lat0)],
            [np.cos(lon0), -np.sin(lon0) * np.sin(lat0), np.sin(lon0) * np.cos(lat0)],
            [0, np.cos(lat0), np.sin(lat0)],
        ]
    ).inv()
    enu_rot = r_ecef_to_enu * r_ecef
    enu_euler = enu_rot.as_euler("xyz", degrees=False)

    # STEP 2: Convert from ENU to Unreal coordinates and adjust from FRD to FRU
    x = np.degrees(enu_euler[0])
    y = -np.degrees(enu_euler[1] - np.pi)
    z = -np.degrees(enu_euler[2] - np.pi)
    return carla.Rotation(x, y, z)


def euler2quat(eulers):
    eulers = np.array(eulers)
    if len(eulers.shape) > 1:
        output_shape = (-1, 4)
    else:
        output_shape = (4,)
    eulers = np.atleast_2d(eulers)
    gamma, theta, psi = eulers[:, 0], eulers[:, 1], eulers[:, 2]

    q0 = np.cos(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(
        gamma / 2
    ) * np.sin(theta / 2) * np.sin(psi / 2)
    q1 = np.sin(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(
        gamma / 2
    ) * np.sin(theta / 2) * np.sin(psi / 2)
    q2 = np.cos(gamma / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(
        gamma / 2
    ) * np.cos(theta / 2) * np.sin(psi / 2)
    q3 = np.cos(gamma / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(
        gamma / 2
    ) * np.sin(theta / 2) * np.cos(psi / 2)

    quats = np.array([q0, q1, q2, q3]).T
    for i in np.arange(len(quats)):
        if quats[i, 0] < 0:
            quats[i] = -quats[i]  # type: ignore
    return quats.reshape(output_shape)


def euler2rot(eulers):
    return quat2rot(euler2quat(eulers))


def carla_vector_to_ecef(vector: carla.Vector3D):
    # It is not this easy
    return np.array([vector.x, -vector.y, vector.z])
