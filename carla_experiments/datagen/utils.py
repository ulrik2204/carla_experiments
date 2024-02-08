import os
import re
from pathlib import Path

import carla
import cv2
import ffmpeg
import numpy as np
import pymap3d as pm
from scipy.spatial.transform import Rotation as R


def frames_to_video(
    input_folder: Path,
    output_file_hevc: Path,
    fps: int,
    delete_intermediate_mp4_file: bool = True,
):
    """Converts a folder of frames to a video file in h265 format.

    This script requires H265 to be installed. On ubuntu:
    sudo apt-get install ffmpeg x265 libx265-dev.

    Args:
        input_folder (str): The folder of frames to convert.
        output_file (str): The output file path, ending in .hevc.
        fps (int): The frame rate of the video.
        use_ffmpeg (bool, optional): If set to true the function will generate a
            .mp4 file with opencv and convert it to h265 with ffmpeg. Defaults to False.
    """
    # Get all files in the input folder

    # Get all file names in the folder
    output_file_hevc = output_file_hevc.with_suffix(".hevc")
    files = [f for f in input_folder.iterdir() if f.is_file()]

    # Sort files based on frame number
    files.sort(key=lambda f: int(os.path.splitext(re.sub("\D", "", f.name))[0]))

    # Read the first frame to determine the size
    frame = cv2.imread(files[0].as_posix())
    height, width, layers = frame.shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # Use 'mp4v' for MP4 format
    output_file_mp4 = output_file_hevc.with_suffix(".mp4")
    out = cv2.VideoWriter(output_file_mp4.as_posix(), fourcc, fps, size)

    # Write each frame to the video
    for file in files:
        frame = cv2.imread(file.as_posix())
        out.write(frame)

    # Release everything when done
    out.release()

    # Convert the video to HEVC format using ffmpeg
    ffmpeg.input(output_file_mp4.as_posix()).output(
        output_file_hevc.as_posix(), vcodec="libx265"
    ).run()
    # delete the mp4 file
    if delete_intermediate_mp4_file:
        output_file_mp4.unlink()


def euler_to_quaternion(rotation: carla.Rotation):
    # I have confirmed that this works the same way OP-Deepdive does it
    # Convert degrees to radians

    pitch = -np.radians(rotation.pitch)
    yaw = -np.radians(rotation.yaw) + np.pi / 2
    roll = np.radians(rotation.roll)

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


def euler_to_quaternion2(map: carla.Map, rotation: carla.Rotation):
    origin_geolocation = map.transform_to_geolocation(carla.Location(0, 0, 0))
    lat0 = origin_geolocation.latitude
    lon0 = origin_geolocation.longitude
    # alt0 = origin_geolocation.altitude

    # I know this rot matrix works in the original implementation
    # Construct rotation matrix from ECEF to ENU frames
    # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    # r = R.from_matrix([
    #     [-np.sin(lon0),  -np.cos(lon0)*np.sin(lat0),   np.cos(lon0)*np.cos(lat0)],
    #     [np.cos(lon0),   -np.sin(lon0)*np.sin(lat0),   np.sin(lon0)*np.cos(lat0)],
    #     [0,               np.cos(lat0),                np.sin(lat0)             ]]).inv()

    # I think we should remove .inv() because I think original example was from ECEF to ENU
    # TODO: Add conversion of alt0 as well (Ulrik)
    r = R.from_matrix(
        [
            [-np.sin(lon0), -np.cos(lon0) * np.sin(lat0), np.cos(lon0) * np.cos(lat0)],
            [np.cos(lon0), -np.sin(lon0) * np.sin(lat0), np.sin(lon0) * np.cos(lat0)],
            [0, np.cos(lat0), np.sin(lat0)],
        ]
    )
    # https://carla.readthedocs.io/en/latest/python_api/#carlarotation
    # https://se.mathworks.com/help/uav/ug/coordinate-systems-for-unreal-engine-simulation-in-uav-toolbox.html
    # In carla
    # pitch is right-handed rotation around the y-axis
    # yaw is left-handed rotation around the z-axis
    # roll is right-handed rotation around the x-axis
    # To adjust for the difference in coordinate systems, we need to negate pitch and yaw
    x = np.radians(rotation.roll)
    y = -np.radians(rotation.pitch)
    z = -np.radians(rotation.yaw)

    # Not tested
    enu_rot = R.from_euler("xyz", (x, y, z), degrees=False)
    ecef_rot = enu_rot.as_matrix() @ r.as_matrix()
    # This returns the quaternion on the [x, y, z, w] format
    ecef_orientation = R.from_matrix(ecef_rot).as_quat()
    x = ecef_orientation[0]
    y = ecef_orientation[1]
    z = ecef_orientation[2]
    w = ecef_orientation[3]

    # print(f"{ecef_orientation = }")
    return np.array([w, x, y, z])


def carla_location_to_ecef2(map: carla.Map, location: carla.Location):
    origin_geolocation = map.transform_to_geolocation(carla.Location(0, 0, 0))
    lat0 = np.radians(origin_geolocation.latitude)
    lon0 = np.radians(origin_geolocation.longitude)
    alt0 = np.radians(origin_geolocation.altitude)
    x0, y0, z0 = pm.geodetic2ecef(lat0, lon0, alt0, deg=False)
    r = R.from_matrix(
        [
            [-np.sin(lon0), -np.cos(lon0) * np.sin(lat0), np.cos(lon0) * np.cos(lat0)],
            [np.cos(lon0), -np.sin(lon0) * np.sin(lat0), np.sin(lon0) * np.cos(lat0)],
            [0, np.cos(lat0), np.sin(lat0)],
        ]
    )
    enu_pos = np.array([location.x, -location.y, location.z])
    ecef_pos = enu_pos @ r.as_matrix() + (x0, y0, z0)
    return ecef_pos


def something():
    # Does not do anything in the gist
    # lat0, lon0, _ = pymap3d.ecef2geodetic(x0, y0, z0, deg=False)

    # Origin of the ENU frame
    lat0, lon0 = (63.421639, 10.428341)
    x0, y0, z0 = pm.geodetic2ecef(lat0, lon0, 0)
    print(x0, y0, z0)

    # I know this rot matrix works in the original implementation
    # Construct rotation matrix from ECEF to ENU frames
    # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    # r = R.from_matrix([
    #     [-np.sin(lon0),  -np.cos(lon0)*np.sin(lat0),   np.cos(lon0)*np.cos(lat0)],
    #     [np.cos(lon0),   -np.sin(lon0)*np.sin(lat0),   np.sin(lon0)*np.cos(lat0)],
    #     [0,               np.cos(lat0),                np.sin(lat0)             ]]).inv()

    # I think we should remove .inv() because I think original example was from ECEF to ENU
    # TODO: Add conversion of alt0 as well (Ulrik)
    r = R.from_matrix(
        [
            [-np.sin(lon0), -np.cos(lon0) * np.sin(lat0), np.cos(lon0) * np.cos(lat0)],
            [np.cos(lon0), -np.sin(lon0) * np.sin(lat0), np.sin(lon0) * np.cos(lat0)],
            [0, np.cos(lat0), np.sin(lat0)],
        ]
    )

    # Not tested
    enu_pos = np.array([0, 0, 0])
    ecef_pos = enu_pos @ r.as_matrix() + (x0, y0, z0)

    # Not tested
    enu_rot = R.from_euler("xyz", (0, 0, 0), degrees=False)
    ecef_rot = enu_rot.as_matrix() @ r.as_matrix()
    ecef_orientation = R.from_matrix(ecef_rot).as_quat()

    print(f"{ecef_pos = }")
    print(f"{ecef_orientation = }")


def carla_location_to_ecef(map: carla.Map, location: carla.Location):
    # I want to check if this finds the origin
    origin_geolocation = map.transform_to_geolocation(carla.Location(0, 0, 0))
    # geolocation = map.transform_to_geolocation(location)
    # https://docs.unrealengine.com/4.27/en-US/BuildingWorlds/Georeferencing/
    e_enu = location.x
    n_enu = -location.y
    u_enu = location.z

    x, y, z = pm.enu2ecef(
        e_enu,
        n_enu,  # Negate y to convert from right-handed to left-handed coordinate system
        u_enu,
        origin_geolocation.latitude,
        origin_geolocation.longitude,
        origin_geolocation.altitude,
        deg=True,
    )
    # latitude = geolocation.latitude
    # longitude = geolocation.longitude
    # altitude = geolocation.altitude

    return np.array([x, y, z])


def ecef_to_carla_location(map: carla.Map, ecef: np.ndarray):
    origin_geolocation = map.transform_to_geolocation(carla.Location(0, 0, 0))
    origin_lat = origin_geolocation.latitude
    origin_long = origin_geolocation.longitude
    origin_alt = origin_geolocation.altitude

    x, y, z = pm.ecef2enu(
        ecef[0], ecef[1], ecef[2], origin_lat, origin_long, origin_alt
    )
    return carla.Location(x, y, z)
