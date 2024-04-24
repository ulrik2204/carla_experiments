import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cycler import cycler
from matplotlib.axes import Axes
from torch import nn

from carla_experiments.common.position_and_rotation import euler2rot, quat2rot

FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array(
    [[FOCAL, 0.0, W / 2.0], [0.0, FOCAL, H / 2.0], [0.0, 0.0, 1.0]]
)

# aka 'K_inv' aka view_frame_from_camera_frame
eon_intrinsics_inv = np.linalg.inv(eon_intrinsics)

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)
view_frame_from_device_frame = device_frame_from_view_frame.T
view_frame_from_device_frame = device_frame_from_view_frame.T


def get_calib_from_vp(vp):
    vp_norm = normalize(vp)
    yaw_calib = np.arctan(vp_norm[0])
    pitch_calib = np.arctan(vp_norm[1] * np.cos(yaw_calib))
    # TODO should be, this but written
    # to be compatible with meshcalib and
    # get_view_frame_from_road_fram
    # pitch_calib = -np.arctan(vp_norm[1]*np.cos(yaw_calib))
    roll_calib = 0
    return roll_calib, pitch_calib, yaw_calib


# aka 'extrinsic_matrix'
# road : x->forward, y -> left, z->up
def get_view_frame_from_road_frame(roll, pitch, yaw, height):
    # TODO
    # calibration pitch is currently defined
    # opposite to pitch in device frame
    pitch = -pitch
    device_from_road = euler2rot([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
    view_from_road = view_frame_from_device_frame.dot(device_from_road)
    return np.hstack((view_from_road, [[0], [height], [0]]))


def vp_from_ke(m):
    """
    Computes the vanishing point from the product of the intrinsic and extrinsic
    matrices C = KE.

    The vanishing point is defined as lim x->infinity C (x, 0, 0, 1).T
    """
    return (m[0, 0] / m[2, 0], m[1, 0] / m[2, 0])


def roll_from_ke(m):
    # note: different from calibration.h/RollAnglefromKE: i think that one's just wrong
    return np.arctan2(
        -(m[1, 0] - m[1, 1] * m[2, 0] / m[2, 1]),
        -(m[0, 0] - m[0, 1] * m[2, 0] / m[2, 1]),
    )


def normalize(img_pts):
    # normalizes image coordinates
    # accepts single pt or array of pts
    img_pts = np.array(img_pts)
    input_shape = img_pts.shape
    img_pts = np.atleast_2d(img_pts)
    img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0], 1))))
    img_pts_normalized = eon_intrinsics_inv.dot(img_pts.T).T
    img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
    return img_pts_normalized[:, :2].reshape(input_shape)


def denormalize(img_pts):
    # denormalizes image coordinates
    # accepts single pt or array of pts
    img_pts = np.array(img_pts)
    input_shape = img_pts.shape
    img_pts = np.atleast_2d(img_pts)
    img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0], 1))))
    img_pts_denormalized = eon_intrinsics.dot(img_pts.T).T
    img_pts_denormalized[img_pts_denormalized[:, 0] > W] = np.nan
    img_pts_denormalized[img_pts_denormalized[:, 0] < 0] = np.nan
    img_pts_denormalized[img_pts_denormalized[:, 1] > H] = np.nan
    img_pts_denormalized[img_pts_denormalized[:, 1] < 0] = np.nan
    return img_pts_denormalized[:, :2].reshape(input_shape)


def device_from_ecef(pos_ecef, orientation_ecef, pt_ecef):
    # device from ecef frame
    # device frame is x -> forward, y-> right, z -> down
    # accepts single pt or array of pts
    input_shape = pt_ecef.shape
    pt_ecef = np.atleast_2d(pt_ecef)
    ecef_from_device_rot = quat2rot(orientation_ecef)
    device_from_ecef_rot = ecef_from_device_rot.T
    pt_ecef_rel = pt_ecef - pos_ecef
    pt_device = np.einsum("jk,ik->ij", device_from_ecef_rot, pt_ecef_rel)
    return pt_device.reshape(input_shape)


def img_from_device(pt_device):
    # img coordinates from pts in device frame
    # first transforms to view frame, then to img coords
    # accepts single pt or array of pts
    input_shape = pt_device.shape
    pt_device = np.atleast_2d(pt_device)
    pt_view = np.einsum("jk,ik->ij", view_frame_from_device_frame, pt_device)

    # This function should never return negative depths
    pt_view[pt_view[:, 2] < 0] = np.nan

    pt_img = pt_view / pt_view[:, 2:3]
    return pt_img.reshape(input_shape)[:, :2]


matplotlib.rcParams["axes.prop_cycle"] = cycler(
    "color",
    [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
)


def draw_trajectory_on_ax(
    ax: Axes,
    trajectories,
    confs,
    line_type="o-",
    transparent=True,
    xlim=(-30, 30),
    ylim=(0, 100),
):
    """
    ax: matplotlib.axes.Axes, the axis to draw trajectories on
    trajectories: List of numpy arrays of shape (num_points, 2 or 3)
    confs: List of numbers, 1 means gt
    """

    # get the max conf
    max_conf = max([conf for conf in confs if conf != 1])

    for idx, (trajectory, conf) in enumerate(zip(trajectories, confs)):
        label = "gt" if conf == 1 else "pred%d (%.3f)" % (idx, conf)
        alpha = 1.0
        if transparent:
            alpha = 1.0 if conf == max_conf else np.clip(conf, 0.1, None)
        plot_args = dict(label=label, alpha=alpha, linewidth=2 if alpha == 1.0 else 1)
        if label == "gt":
            plot_args["color"] = "#d62728"
        ax.plot(
            trajectory[:, 1],  # - for nuscenes and + for comma 2k19
            trajectory[:, 0],
            line_type,
            **plot_args
        )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()

    return ax


def get_val_metric(pred_cls, pred_trajectory, labels, namespace="val"):
    rtn_dict = dict()
    bs, M, num_pts, _ = pred_trajectory.shape

    # Lagecy metric: Prediction L2 loss
    pred_label = torch.argmax(pred_cls, -1)  # B,
    pred_trajectory_single = pred_trajectory[
        torch.tensor(range(bs), device=pred_cls.device), pred_label, ...
    ]
    l2_dists = F.mse_loss(
        pred_trajectory_single, labels, reduction="none"
    )  # B, num_pts, 2 or 3

    # Lagecy metric: cls Acc
    gt_trajectory_M = labels[:, None, ...].expand(-1, M, -1, -1)
    l2_distances = F.mse_loss(pred_trajectory, gt_trajectory_M, reduction="none").sum(
        dim=(2, 3)
    )  # B, M
    best_match = torch.argmin(l2_distances, -1)  # B,
    rtn_dict.update(
        {"l2_dist": l2_dists.mean(dim=(1, 2)), "cls_acc": best_match == pred_label}
    )

    # New Metric
    distance_splits = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 1000))
    AP_thresholds = (0.5, 1, 2)
    euclidean_distances = l2_dists.sum(
        -1
    ).sqrt()  # euclidean distances over the points: [B, num_pts]
    x_distances = labels[..., 0]  # B, num_pts

    for min_dst, max_dst in distance_splits:
        points_mask = (x_distances >= min_dst) & (x_distances < max_dst)  # B, num_pts,
        if points_mask.sum() == 0:
            continue  # No gt points in this range
        rtn_dict.update(
            {"eucliden_%d_%d" % (min_dst, max_dst): euclidean_distances[points_mask]}
        )  # [sum(mask), ]
        rtn_dict.update(
            {
                "eucliden_x_%d_%d"
                % (min_dst, max_dst): l2_dists[..., 0][points_mask].sqrt()
            }
        )  # [sum(mask), ]
        rtn_dict.update(
            {
                "eucliden_y_%d_%d"
                % (min_dst, max_dst): l2_dists[..., 1][points_mask].sqrt()
            }
        )  # [sum(mask), ]

        for AP_threshold in AP_thresholds:
            hit_mask = (euclidean_distances < AP_threshold) & points_mask
            rtn_dict.update(
                {
                    "AP_%d_%d_%s"
                    % (min_dst, max_dst, AP_threshold): hit_mask[points_mask]
                }
            )

    # add namespace
    if namespace is not None:
        for k in list(rtn_dict.keys()):
            rtn_dict["%s/%s" % (namespace, k)] = rtn_dict.pop(k)
    return rtn_dict


def get_val_metric_keys(namespace="val"):
    rtn_dict = dict()
    rtn_dict.update({"l2_dist": [], "cls_acc": []})

    # New Metric
    distance_splits = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 1000))
    AP_thresholds = (0.5, 1, 2)

    for min_dst, max_dst in distance_splits:
        rtn_dict.update({"eucliden_%d_%d" % (min_dst, max_dst): []})  # [sum(mask), ]
        rtn_dict.update({"eucliden_x_%d_%d" % (min_dst, max_dst): []})  # [sum(mask), ]
        rtn_dict.update({"eucliden_y_%d_%d" % (min_dst, max_dst): []})  # [sum(mask), ]
        for AP_threshold in AP_thresholds:
            rtn_dict.update({"AP_%d_%d_%s" % (min_dst, max_dst, AP_threshold): []})

    # add namespace
    if namespace is not None:
        for k in list(rtn_dict.keys()):
            rtn_dict["%s/%s" % (namespace, k)] = rtn_dict.pop(k)
    return rtn_dict


def generate_random_params_for_warp(img, random_rate=0.1):
    h, w = img.shape[:2]

    width_max = random_rate * w
    height_max = random_rate * h

    # 8 offsets
    w_offsets = list(np.random.uniform(0, width_max) for _ in range(4))
    h_offsets = list(np.random.uniform(0, height_max) for _ in range(4))

    return w_offsets, h_offsets


def warp(img, w_offsets, h_offsets):
    h, w = img.shape[:2]

    original_corner_pts = np.array(
        (
            (w_offsets[0], h_offsets[0]),
            (w - w_offsets[1], h_offsets[1]),
            (w_offsets[2], h - h_offsets[2]),
            (w - w_offsets[3], h - h_offsets[3]),
        ),
        dtype=np.float32,
    )

    target_corner_pts = np.array(
        (
            (0, 0),  # Top-left
            (w, 0),  # Top-right
            (0, h),  # Bottom-left
            (w, h),  # Bottom-right
        ),
        dtype=np.float32,
    )

    transform_matrix = cv2.getPerspectiveTransform(
        original_corner_pts, target_corner_pts
    )

    transformed_image = cv2.warpPerspective(img, transform_matrix, (w, h))

    return transformed_image


def draw_path(
    device_path,
    img,
    width=1,
    height=1.2,
    fill_color=(128, 0, 255),
    line_color=(0, 255, 0),
):
    # device_path: N, 3
    device_path_l = device_path + np.array([0, 0, height])
    device_path_r = device_path + np.array([0, 0, height])
    device_path_l[:, 1] -= width
    device_path_r[:, 1] += width

    img_points_norm_l = img_from_device(device_path_l)
    img_points_norm_r = img_from_device(device_path_r)

    img_pts_l = denormalize(img_points_norm_l)
    img_pts_r = denormalize(img_points_norm_r)
    # filter out things rejected along the way
    valid = np.logical_and(
        np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1)
    )
    img_pts_l = img_pts_l[valid].astype(int)
    img_pts_r = img_pts_r[valid].astype(int)

    for i in range(1, len(img_pts_l)):
        u1, v1, u2, v2 = np.append(img_pts_l[i - 1], img_pts_r[i - 1])
        u3, v3, u4, v4 = np.append(img_pts_l[i], img_pts_r[i])
        pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.int32).reshape(
            (-1, 1, 2)
        )
        if fill_color:
            cv2.fillPoly(img, [pts], fill_color)
        if line_color:
            cv2.polylines(img, [pts], True, line_color)
