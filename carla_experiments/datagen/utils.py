import os
import re
from pathlib import Path
from typing import List

import carla
import cv2
import ffmpeg
import numpy as np
from PIL import Image


def pil_images_to_mp4(images: List[Image.Image], output_file: str, fps: int):

    # Convert the first image to set the video properties
    first_image = images[0]
    height = first_image.height
    width = first_image.width

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # 'mp4v' for .mp4 files
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Process each Carla image
    for item_image in images:
        frame = np.array(item_image)
        converted_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(converted_frame)  # Expects bgr format

    # Release the VideoWriter
    video.release()


def carla_image_to_bgr_array(carla_image: carla.Image) -> np.ndarray:
    array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(
        array, (carla_image.height, carla_image.width, 4)
    )  # 4 channels (BGRA)
    array = array[:, :, :3]  # Drop the alpha channel
    return array


def mp4_to_hevc(input_file_mp4: Path, output_file_hevc: Path):
    ffmpeg.input(input_file_mp4.as_posix()).output(
        output_file_hevc.as_posix(), vcodec="libx265"
    ).run()
    # delete the mp4 file


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
