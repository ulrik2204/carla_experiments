from dataclasses import dataclass

import cv2
import numpy as np
import torch

from openpilot_exploration.common.position_and_rotation import euler2rot
from openpilot_exploration.openpilot_common.camera import (
    eon_fcam_intrinsics,
    tici_ecam_intrinsics,
    tici_fcam_intrinsics,
)

FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array(
    [[FOCAL, 0.0, W / 2.0], [0.0, FOCAL, H / 2.0], [0.0, 0.0, 1.0]]
)

# aka 'K_inv' aka view_frame_from_camera_frame
eon_intrinsics_inv = np.linalg.inv(eon_intrinsics)
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


def get_view_frame_from_road_frame(roll, pitch, yaw, height):
    device_from_road = euler2rot([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
    view_from_road = VIEW_FRAME_FROM_DEVICE_FRAME.dot(device_from_road)
    return np.hstack((view_from_road, [[0], [height], [0]]))


# aka 'extrinsic_matrix'
def get_view_frame_from_calib_frame(roll, pitch, yaw, height):
    device_from_calib = euler2rot([roll, pitch, yaw])
    view_from_calib = VIEW_FRAME_FROM_DEVICE_FRAME.dot(device_from_calib)
    return np.hstack((view_from_calib, [[0], [height], [0]]))


# segnet
SEGNET_SIZE = (512, 384)

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


# BIG model
BIGMODEL_INPUT_SIZE = (1024, 512)
BIGMODEL_YUV_SIZE = (BIGMODEL_INPUT_SIZE[0], BIGMODEL_INPUT_SIZE[1] * 3 // 2)

bigmodel_fl = 910.0
bigmodel_intrinsics = np.array(
    [
        [bigmodel_fl, 0.0, 0.5 * BIGMODEL_INPUT_SIZE[0]],
        [0.0, bigmodel_fl, 256 + MEDMODEL_CY],
        [0.0, 0.0, 1.0],
    ]
)


# SBIG model (big model with the size of small model)
SBIGMODEL_INPUT_SIZE = (512, 256)
SBIGMODEL_YUV_SIZE = (SBIGMODEL_INPUT_SIZE[0], SBIGMODEL_INPUT_SIZE[1] * 3 // 2)

sbigmodel_fl = 455.0
sbigmodel_intrinsics = np.array(
    [
        [sbigmodel_fl, 0.0, 0.5 * SBIGMODEL_INPUT_SIZE[0]],
        [0.0, sbigmodel_fl, 0.5 * (256 + MEDMODEL_CY)],
        [0.0, 0.0, 1.0],
    ]
)

bigmodel_frame_from_calib_frame = np.dot(
    bigmodel_intrinsics, get_view_frame_from_calib_frame(0, 0, 0, 0)
)


sbigmodel_frame_from_calib_frame = np.dot(
    sbigmodel_intrinsics, get_view_frame_from_calib_frame(0, 0, 0, 0)
)

medmodel_frame_from_calib_frame = np.dot(
    medmodel_intrinsics, get_view_frame_from_calib_frame(0, 0, 0, 0)
)

medmodel_frame_from_bigmodel_frame = np.dot(
    medmodel_intrinsics, np.linalg.inv(bigmodel_intrinsics)
)

calib_from_medmodel = np.linalg.inv(medmodel_frame_from_calib_frame[:, :3])
calib_from_sbigmodel = np.linalg.inv(sbigmodel_frame_from_calib_frame[:, :3])


# # This function is verified to give similar results to xx.uncommon.utils.transform_img
# def get_warp_matrix(
#     device_from_calib_euler: np.ndarray,  # extrinsic?
#     intrinsics: np.ndarray,
#     bigmodel_frame: bool = False,
# ) -> np.ndarray:
#     calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
#     device_from_calib = euler2rot(device_from_calib_euler)
#     camera_from_calib = intrinsics @ VIEW_FRAME_FROM_DEVICE_FRAME @ device_from_calib
#     warp_matrix: np.ndarray = camera_from_calib @ calib_from_model
#     return warp_matrix


def get_warp_matrix(
    device_from_calib_euler: np.ndarray,
    wide_camera: bool = False,
    bigmodel_frame: bool = False,
) -> np.ndarray:
    tici = True
    if tici and wide_camera:
        cam_intrinsics = tici_ecam_intrinsics
    elif tici:
        cam_intrinsics = tici_fcam_intrinsics
    else:
        cam_intrinsics = eon_fcam_intrinsics

    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = euler2rot(device_from_calib_euler)
    camera_from_calib = (
        cam_intrinsics @ VIEW_FRAME_FROM_DEVICE_FRAME @ device_from_calib
    )
    warp_matrix: np.ndarray = camera_from_calib @ calib_from_model
    return warp_matrix


def warp_image(
    img: np.ndarray,
    device_from_calib_euler: np.ndarray,
    wide_camera: bool = False,
    bigmodel_frame: bool = False,
    output_size=(512, 256),
) -> np.ndarray:
    warp_matrix = get_warp_matrix(device_from_calib_euler, wide_camera, bigmodel_frame)
    war = cv2.warpPerspective(
        img,
        warp_matrix,
        output_size,
        flags=cv2.WARP_INVERSE_MAP,
    )
    return np.array(war)


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    focal_length: float

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def intrinsics(self):
        # aka 'K' aka camera_frame_from_view_frame
        return np.array(
            [
                [self.focal_length, 0.0, float(self.width) / 2],
                [0.0, self.focal_length, float(self.height) / 2],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def intrinsics_inv(self):
        # aka 'K_inv' aka view_frame_from_camera_frame
        return np.linalg.inv(self.intrinsics)


@dataclass(frozen=True)
class _NoneCameraConfig(CameraConfig):
    width: int = 0
    height: int = 0
    focal_length: float = 0


@dataclass(frozen=True)
class DeviceCameraConfig:
    fcam: CameraConfig
    dcam: CameraConfig
    ecam: CameraConfig

    def all_cams(self):
        for cam in ["fcam", "dcam", "ecam"]:
            if not isinstance(getattr(self, cam), _NoneCameraConfig):
                yield cam, getattr(self, cam)


_ar_ox_fisheye = CameraConfig(
    1928, 1208, 567.0
)  # focal length probably wrong? magnification is not consistent across frame
_os_fisheye = CameraConfig(2688, 1520, 567.0 / 2 * 3)
_ar_ox_config = DeviceCameraConfig(
    CameraConfig(1928, 1208, 2648.0), _ar_ox_fisheye, _ar_ox_fisheye
)
_os_config = DeviceCameraConfig(
    CameraConfig(2688, 1520, 2648.0 * 2 / 3), _os_fisheye, _os_fisheye
)
_neo_config = DeviceCameraConfig(
    CameraConfig(1164, 874, 910.0), CameraConfig(816, 612, 650.0), _NoneCameraConfig()
)

DEVICE_CAMERAS = {
    # A "device camera" is defined by a device type and sensor
    # sensor type was never set on eon/neo/two
    ("neo", "unknown"): _neo_config,
    # unknown here is AR0231, field was added with OX03C10 support
    ("tici", "unknown"): _ar_ox_config,
    # before deviceState.deviceType was set, assume tici AR config
    ("unknown", "ar0231"): _ar_ox_config,
    ("unknown", "ox03c10"): _ar_ox_config,
    # simulator (emulates a tici)
    ("pc", "unknown"): _ar_ox_config,
}

YUV_FROM_RGB = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.14714119, -0.28886916, 0.43601035],
        [0.61497538, -0.51496512, -0.10001026],
    ]
)
RGB_FROM_YUV = np.linalg.inv(YUV_FROM_RGB)


# Openpilot code
def rgb24toyuv(rgb):
    # img is shape (height, width, 3)
    img = np.dot(rgb.reshape(-1, 3), YUV_FROM_RGB.T).reshape(rgb.shape)

    ys = img[:, :, 0]
    us = (
        img[::2, ::2, 1] + img[1::2, ::2, 1] + img[::2, 1::2, 1] + img[1::2, 1::2, 1]
    ) / 4 + 128
    vs = (
        img[::2, ::2, 2] + img[1::2, ::2, 2] + img[::2, 1::2, 2] + img[1::2, 1::2, 2]
    ) / 4 + 128

    return ys, us, vs


# Openpilot code
def rgb24toyuv420(rgb: np.ndarray):
    # rgb is shape (height, width, 3)
    ys, us, vs = rgb24toyuv(rgb)

    y_len = rgb.shape[0] * rgb.shape[1]
    uv_len = y_len // 4

    yuv420 = np.empty(y_len + 2 * uv_len, dtype=rgb.dtype)
    yuv420[:y_len] = ys.reshape(-1)
    yuv420[y_len : y_len + uv_len] = us.reshape(-1)
    yuv420[y_len + uv_len : y_len + 2 * uv_len] = vs.reshape(-1)

    return yuv420.clip(0, 255).astype("uint8")


def yuvtorgb24(ys, us, vs):
    # Adjust U and V channels by subtracting 128
    us = us - 128
    vs = vs - 128

    # Upsample U and V to the size of Y
    u_upsampled = np.repeat(np.repeat(us, 2, axis=0), 2, axis=1)
    v_upsampled = np.repeat(np.repeat(vs, 2, axis=0), 2, axis=1)

    # Stack the Y, U, and V channels to form a YUV image
    yuv = np.stack((ys, u_upsampled, v_upsampled), axis=-1)

    # Reshape to match the input image's shape
    yuv = yuv.reshape(-1, 3)

    # Convert YUV back to RGB
    rgb = np.dot(yuv, RGB_FROM_YUV.T)

    # Clip values to valid range
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Reshape to the original image shape
    rgb = rgb.reshape(ys.shape[0], ys.shape[1], 3)

    return rgb


def yuv420torgb24(yuv420, width, height):
    # yuv420 is a flat array of Y, U, and V values

    y_len = height * width
    uv_len = y_len // 4

    ys = yuv420[:y_len].reshape((height, width))
    us = yuv420[y_len : y_len + uv_len].reshape((height // 2, width // 2))
    vs = yuv420[y_len + uv_len : y_len + 2 * uv_len].reshape((height // 2, width // 2))

    rgb = yuvtorgb24(ys, us, vs)

    return rgb


# custom
def rgb_to_6_channel_yuv(frames: torch.Tensor) -> torch.Tensor:
    """Convert a RGB image to a 6-channel YUV image.

    Args:
        frames (torch.Tensor): A tensor of shape (num_frames, height, width, 3)
            where the last dimension is the RGB channels.

    Returns:
        torch.Tensor: A tensor of shape (num_frames, height//2, width//2, 6)
            being a 6 channel YUV image where the first 4 channels are the Y channels,
            and the last 2 are the U and V channels at half resolution.
    """
    num_frames, height, width, _ = frames.shape

    # Prepare output tensor
    output = torch.zeros(
        (num_frames, height // 2, width // 2, 6),
        dtype=frames.dtype,
        device=frames.device,
    )
    # yuv_from_rgb = torch.tensor(YUV_FROM_RGB, device=frames.device, dtype=torch.float32)

    # Process each frame
    for i in range(num_frames):
        # Split Y, U, V channels
        frame = frames[i]
        yuv_frame = torch.tensor(
            cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_RGB2YUV),
            device=frames.device,
            dtype=frames.dtype,
        )
        Y = yuv_frame[:, :, 0]
        U = yuv_frame[:, :, 1]
        V = yuv_frame[:, :, 2]

        # Channel 0: Y[::2, ::2]
        output[i, :, :, 0] = Y[::2, ::2]
        # Channel 1: Y[::2, 1::2]
        output[i, :, :, 1] = Y[::2, 1::2]
        # Channel 2: Y[1::2, ::2]
        output[i, :, :, 2] = Y[1::2, ::2]
        # Channel 3: Y[1::2, 1::2]
        output[i, :, :, 3] = Y[1::2, 1::2]

        # Channel 4: U - at half resolution
        output[i, :, :, 4] = U[::2, ::2]
        # Channel 5: V - at half resolution
        output[i, :, :, 5] = V[::2, ::2]

    return output


def yuv_6_channel_to_rgb(frames: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of video frames from 6-channel YUV format back to standard 3-channel YUV format.

    Args:
        frames (torch.Tensor): A tensor of shape (num_frames, height, width, 6)
            where the last dimension represents the 6-channel YUV.

    Returns:
        torch.Tensor: A tensor of shape (num_frames, height*2, width*2, 3)
    """
    num_frames, height, width, _ = frames.shape

    # Prepare output tensor
    rgb_frames = []
    # rgb_from_yuv = torch.tensor(RGB_FROM_YUV, device=frames.device, dtype=torch.float32)

    # Process each frame
    for i in range(num_frames):
        # Retrieve all channels
        Y00 = frames[i, :, :, 0]
        Y01 = frames[i, :, :, 1]
        Y10 = frames[i, :, :, 2]
        Y11 = frames[i, :, :, 3]
        U = frames[i, :, :, 4]
        V = frames[i, :, :, 5]
        yuv_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

        # Reconstruct full resolution Y channel
        yuv_image[::2, ::2, 0] = Y00
        yuv_image[::2, 1::2, 0] = Y01
        yuv_image[1::2, ::2, 0] = Y10
        yuv_image[1::2, 1::2, 0] = Y11

        # Reconstruct full resolution U and V channels
        U_upsampled = U.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        V_upsampled = V.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

        # Fix to properly fit the tensor dimensions
        yuv_image[:, :, 1] = U_upsampled[: height * 2, : width * 2]
        yuv_image[:, :, 2] = V_upsampled[: height * 2, : width * 2]
        rgb = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
        rgb_frames.append(torch.tensor(rgb, dtype=frames.dtype, device=frames.device))

    return torch.stack(rgb_frames)


def car_space_to_bb(x, y, z, rpy, intrinsic, width=1928, height=1208):
    car_space_projective = np.column_stack((x, y, z)).T
    ext = get_view_frame_from_calib_frame(rpy[0], rpy[1], rpy[2], 0.0)[:, :3]
    ep = ext.dot(car_space_projective)
    kep = intrinsic.dot(ep)
    values = (kep[:-1, :] / kep[-1, :]).T
    values[values[:, 0] > width] = np.nan
    values[values[:, 0] < 0] = np.nan
    values[values[:, 1] > width] = np.nan
    values[values[:, 1] < 0] = np.nan
    # values = np.round(values).astype(int)
    return values
