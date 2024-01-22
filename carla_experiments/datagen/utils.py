import os
import re
from pathlib import Path

import carla
import cv2
import ffmpeg
import numpy as np


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
    # Convert degrees to radians
    pitch = np.radians(rotation.pitch)
    yaw = np.radians(rotation.yaw)
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

    return np.array([w, x, y, z])
