"""This code is from https://github.com/OpenDriveLab/Openpilot-Deepdive/blob/main/data.py"""

import json
import os

import capnp
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from torchvision import transforms

from carla_experiments.common.position_and_rotation import quat2rot

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
device_frame_from_view_frame = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)
view_frame_from_device_frame = device_frame_from_view_frame.T

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


def calibration(extrinsic_matrix, cam_intrinsics, device_frame_from_road_frame=None):
    if device_frame_from_road_frame is None:
        device_frame_from_road_frame = np.hstack(
            (np.diag([1, -1, -1]), [[0], [0], [1.51]])
        )
    med_frame_from_ground = (
        medmodel_intrinsics
        @ view_frame_from_device_frame
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


class PlanningDataset(Dataset):
    def __init__(self, root="data", json_path_pattern="p3_%s.json", split="train"):
        self.samples = json.load(open(os.path.join(root, json_path_pattern % split)))
        print(
            "PlanningDataset: %d samples loaded from %s"
            % (len(self.samples), os.path.join(root, json_path_pattern % split))
        )
        self.split = split

        self.img_root = os.path.join(root, "nuscenes")
        self.transforms = transforms.Compose(
            [
                # transforms.Resize((900 // 2, 1600 // 2)),
                # transforms.Resize((9 * 32, 16 * 32)),
                transforms.Resize((128, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.3890, 0.3937, 0.3851], [0.2172, 0.2141, 0.2209]
                ),
            ]
        )

        self.enable_aug = False
        self.view_transform = False

    def _get_cv2_image(self, path):
        return cv2.imread(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs, future_poses = sample["imgs"], sample["future_poses"]

        # process future_poses
        future_poses = torch.tensor(future_poses)
        future_poses[:, 0] = future_poses[:, 0].clamp(
            1e-2,
        )  # the car will never go backward

        imgs = list(self._get_cv2_image(os.path.join(self.img_root, p)) for p in imgs)
        imgs = list(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs)  # RGB

        # process images
        if self.enable_aug and self.split == "train":
            # data augumentation when training
            # random distort (warp)
            w_offsets, h_offsets = generate_random_params_for_warp(
                imgs[0], random_rate=0.1
            )
            imgs = list(warp(img, w_offsets, h_offsets) for img in imgs)

            # random flip
            if np.random.rand() > 0.5:
                imgs = list(img[:, ::-1, :] for img in imgs)
                future_poses[:, 1] *= -1

        if self.view_transform:
            camera_rotation_matrix = np.linalg.inv(
                np.array(sample["camera_rotation_matrix_inv"])
            )
            camera_translation = -np.array(sample["camera_translation_inv"])
            camera_extrinsic = np.vstack(
                (
                    np.hstack(
                        (camera_rotation_matrix, camera_translation.reshape((3, 1)))
                    ),
                    np.array([0, 0, 0, 1]),
                )
            )
            camera_extrinsic = np.linalg.inv(camera_extrinsic)
            warp_matrix = calibration(
                camera_extrinsic, np.array(sample["camera_intrinsic"])
            )
            imgs = list(
                cv2.warpPerspective(
                    src=img, M=warp_matrix, dsize=(256, 128), flags=cv2.WARP_INVERSE_MAP
                )
                for img in imgs
            )

        # cvt back to PIL images
        # cv2.imshow('0', imgs[0])
        # cv2.imshow('1', imgs[1])
        # cv2.waitKey(0)
        imgs = list(Image.fromarray(img) for img in imgs)
        imgs = list(self.transforms(img) for img in imgs)
        input_img = torch.cat(imgs, dim=0)

        return dict(
            input_img=input_img,
            future_poses=future_poses,
            camera_intrinsic=torch.tensor(sample["camera_intrinsic"]),
            camera_extrinsic=torch.tensor(sample["camera_extrinsic"]),
            camera_translation_inv=torch.tensor(sample["camera_translation_inv"]),
            camera_rotation_matrix_inv=torch.tensor(
                sample["camera_rotation_matrix_inv"]
            ),
        )


class SequencePlanningDataset(PlanningDataset):
    def __init__(self, root="data", json_path_pattern="p3_%s.json", split="train"):
        print("Sequence", end="")
        self.fix_seq_length = 18
        super().__init__(root=root, json_path_pattern=json_path_pattern, split=split)

    def __getitem__(self, idx):
        seq_samples = self.samples[idx]
        seq_length = len(seq_samples)
        if seq_length < self.fix_seq_length:
            # Only 1 sample < 28 (==21)
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        if seq_length > self.fix_seq_length:
            seq_length_delta = seq_length - self.fix_seq_length
            seq_length_delta = np.random.randint(0, seq_length_delta + 1)
            seq_samples = seq_samples[
                seq_length_delta : self.fix_seq_length + seq_length_delta
            ]

        seq_future_poses = list(smp["future_poses"] for smp in seq_samples)
        seq_imgs = list(smp["imgs"] for smp in seq_samples)

        seq_input_img = []
        for imgs in seq_imgs:
            imgs = list(
                self._get_cv2_image(os.path.join(self.img_root, p)) for p in imgs
            )
            imgs = list(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs)  # RGB
            imgs = list(Image.fromarray(img) for img in imgs)
            imgs = list(self.transforms(img) for img in imgs)
            input_img = torch.cat(imgs, dim=0)
            seq_input_img.append(input_img[None])
        seq_input_img = torch.cat(seq_input_img)

        return dict(
            seq_input_img=seq_input_img,  # torch.Size([28, 10, 3])
            seq_future_poses=torch.tensor(
                seq_future_poses
            ),  # torch.Size([28, 6, 128, 256])
            camera_intrinsic=torch.tensor(seq_samples[0]["camera_intrinsic"]),
            camera_extrinsic=torch.tensor(seq_samples[0]["camera_extrinsic"]),
            camera_translation_inv=torch.tensor(
                seq_samples[0]["camera_translation_inv"]
            ),
            camera_rotation_matrix_inv=torch.tensor(
                seq_samples[0]["camera_rotation_matrix_inv"]
            ),
        )


class Comma2k19SequenceDataset(PlanningDataset):
    def __init__(
        self, split_txt_path, prefix, mode, use_memcache=True, return_origin=False
    ):
        self.split_txt_path = split_txt_path
        self.prefix = prefix

        self.samples = open(split_txt_path).readlines()
        self.samples = [i.strip() for i in self.samples]

        assert mode in ("train", "val", "demo")
        self.mode = mode
        if self.mode == "demo":
            print("Comma2k19SequenceDataset: DEMO mode is on.")

        self.fix_seq_length = 800 if mode == "train" else 800

        self.transforms = transforms.Compose(
            [
                # transforms.Resize((900 // 2, 1600 // 2)),
                # transforms.Resize((9 * 32, 16 * 32)),
                transforms.Resize((128, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.3890, 0.3937, 0.3851], [0.2172, 0.2141, 0.2209]
                ),
            ]
        )

        self.warp_matrix = calibration(
            extrinsic_matrix=np.array(
                [[0, -1, 0, 0], [0, 0, -1, 1.22], [1, 0, 0, 0], [0, 0, 0, 1]]
            ),
            cam_intrinsics=np.array([[910, 0, 582], [0, 910, 437], [0, 0, 1]]),
            device_frame_from_road_frame=np.hstack(
                (np.diag([1, -1, -1]), [[0], [0], [1.22]])
            ),
        )

        self.return_origin = return_origin

        # from OpenPilot
        self.num_pts = 10 * 20  # 10 s * 20 Hz = 200 frames
        self.t_anchors = np.array(
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
        self.t_idx = np.linspace(0, 10, num=self.num_pts)

    def _get_cv2_vid(self, path):
        return cv2.VideoCapture(path)

    def _get_numpy(self, path):
        return np.load(path)

    def __getitem__(self, idx):
        seq_sample_path = self.prefix + self.samples[idx]
        cap = self._get_cv2_vid(seq_sample_path + "/video.hevc")
        if not cap.isOpened():
            raise RuntimeError
        imgs = []  # <--- all frames here
        origin_imgs = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                imgs.append(frame)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                if self.return_origin:
                    origin_imgs.append(frame)
            else:
                break
        cap.release()

        seq_length = len(imgs)

        if self.mode == "demo":
            self.fix_seq_length = seq_length - self.num_pts - 1

        if seq_length < self.fix_seq_length + self.num_pts:
            print(
                "The length of sequence",
                seq_sample_path,
                "is too short",
                "(%d < %d)" % (seq_length, self.fix_seq_length + self.num_pts),
            )
            return self.__getitem__(idx + 1)

        seq_length_delta = seq_length - (self.fix_seq_length + self.num_pts)
        seq_length_delta = np.random.randint(1, seq_length_delta + 1)

        seq_start_idx = seq_length_delta  # often 1
        seq_end_idx = (
            seq_length_delta + self.fix_seq_length
        )  # 800 in train and 1000 in val

        # seq_input_img
        imgs = imgs[seq_start_idx - 1 : seq_end_idx]  # contains one more img
        imgs = [
            cv2.warpPerspective(
                src=img,
                M=self.warp_matrix,
                dsize=(512, 256),
                flags=cv2.WARP_INVERSE_MAP,
            )
            for img in imgs
        ]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        imgs = list(Image.fromarray(img) for img in imgs)
        imgs = list(self.transforms(img)[None] for img in imgs)
        input_img = torch.cat(imgs, dim=0)  # [N+1, 3, H, W]
        del imgs
        input_img = torch.cat((input_img[:-1, ...], input_img[1:, ...]), dim=1)
        print(
            "seq_length",
            seq_length,
            "fix_seq_length",
            self.fix_seq_length,
            "seq_start_idx/seq_length_delta",
            seq_start_idx,
            "seq_end_idx",
            seq_end_idx,
            "self.num_pts",
            self.num_pts,
        )

        # poses
        frame_positions = self._get_numpy(
            self.prefix + self.samples[idx] + "/global_pose/frame_positions"
        )[seq_start_idx : seq_end_idx + self.num_pts]
        frame_orientations = self._get_numpy(
            self.prefix + self.samples[idx] + "/global_pose/frame_orientations"
        )[seq_start_idx : seq_end_idx + self.num_pts]

        future_poses = []
        for i in range(self.fix_seq_length):
            ecef_from_local = quat2rot(frame_orientations[i])
            local_from_ecef = ecef_from_local.T
            frame_positions_local = np.einsum(
                "ij,kj->ki", local_from_ecef, frame_positions - frame_positions[i]
            ).astype(np.float32)

            # Time-Anchor like OpenPilot
            fs = [
                interp1d(self.t_idx, frame_positions_local[i : i + self.num_pts, j])
                for j in range(3)
            ]
            interp_positions = [fs[j](self.t_anchors)[:, None] for j in range(3)]
            interp_positions = np.concatenate(interp_positions, axis=1)

            future_poses.append(interp_positions)
        future_poses = torch.tensor(np.array(future_poses), dtype=torch.float32)

        rtn_dict = dict(
            seq_input_img=input_img,  # torch.Size([N, 6, 128, 256])
            seq_future_poses=future_poses,  # torch.Size([N, num_pts, 3])
            # camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
            # camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
            # camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
            # camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
        )

        # For DEMO
        if self.return_origin:
            origin_imgs = origin_imgs[seq_start_idx:seq_end_idx]
            origin_imgs = [
                torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None]
                for img in origin_imgs
            ]
            origin_imgs = torch.cat(origin_imgs, dim=0)  # N, H_ori, W_ori, 3
            rtn_dict["origin_imgs"] = origin_imgs

        return rtn_dict
