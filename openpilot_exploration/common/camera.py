import numpy as np

## -- hardcoded hardware params --
eon_f_focal_length = 910.0
eon_d_focal_length = 650.0
tici_f_focal_length = 2648.0
tici_e_focal_length = tici_d_focal_length = (
    567.0  # probably wrong? magnification is not consistent across frame
)

eon_f_frame_size = (1164, 874)
eon_d_frame_size = (816, 612)
tici_f_frame_size = tici_e_frame_size = tici_d_frame_size = (1928, 1208)

# aka 'K' aka camera_frame_from_view_frame
eon_fcam_intrinsics = np.array(
    [
        [eon_f_focal_length, 0.0, float(eon_f_frame_size[0]) / 2],
        [0.0, eon_f_focal_length, float(eon_f_frame_size[1]) / 2],
        [0.0, 0.0, 1.0],
    ]
)
eon_intrinsics = eon_fcam_intrinsics  # xx

eon_dcam_intrinsics = np.array(
    [
        [eon_d_focal_length, 0.0, float(eon_d_frame_size[0]) / 2],
        [0.0, eon_d_focal_length, float(eon_d_frame_size[1]) / 2],
        [0.0, 0.0, 1.0],
    ]
)

tici_fcam_intrinsics = np.array(
    [
        [tici_f_focal_length, 0.0, float(tici_f_frame_size[0]) / 2],
        [0.0, tici_f_focal_length, float(tici_f_frame_size[1]) / 2],
        [0.0, 0.0, 1.0],
    ]
)

tici_dcam_intrinsics = np.array(
    [
        [tici_d_focal_length, 0.0, float(tici_d_frame_size[0]) / 2],
        [0.0, tici_d_focal_length, float(tici_d_frame_size[1]) / 2],
        [0.0, 0.0, 1.0],
    ]
)

tici_ecam_intrinsics = tici_dcam_intrinsics

# aka 'K_inv' aka view_frame_from_camera_frame
eon_fcam_intrinsics_inv = np.linalg.inv(eon_fcam_intrinsics)
eon_intrinsics_inv = eon_fcam_intrinsics_inv  # xx

tici_fcam_intrinsics_inv = np.linalg.inv(tici_fcam_intrinsics)
tici_ecam_intrinsics_inv = np.linalg.inv(tici_ecam_intrinsics)


FULL_FRAME_SIZE = tici_f_frame_size
FOCAL = tici_f_focal_length
fcam_intrinsics = tici_fcam_intrinsics

W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]


# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)
view_frame_from_device_frame = device_frame_from_view_frame.T
