import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch

from carla_experiments.common.constants import (
    SUPERCOMBO_META_SLICES,
    SUPERCOMBO_OUTPUT_SLICES,
    SUPERCOMBO_PLAN_SLICES,
)
from carla_experiments.common.openpilot_repo import download_github_file
from carla_experiments.common.types_common import (
    FirstSliceSupercomboOutput,
    MetaSliced,
    MetaTensors,
    PlanFull,
    PlanSliced,
    PlanTensors,
    PoseTensors,
    SupercomboFullOutput,
    SupercomboOutputLogged,
)


def get_supercombo_onnx_model(path: Path) -> ort.InferenceSession:
    if not path.exists():
        download_github_file(
            repo_owner="commaai",
            repo_name="openpilot",
            file_path_in_repo="selfdrive/modeld/models/supercombo.onnx",
            save_path=path,
            saved_at="git-lfs",
            main_branch_name="master",
        )
    return create_ort_session(path.as_posix(), fp16_to_fp32=True)


def torch_dict_to_numpy(inputs, dtype=np.float32) -> Dict[str, np.ndarray]:
    result = dict()
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu().numpy().astype(dtype)
        elif isinstance(v, dict):
            result[k] = torch_dict_to_numpy(v)
        else:
            raise ValueError(f"Unsupported type {type(v)}")
    return result


# def sigmoid(x: torch.Tensor):
#     return 1.0 / (1.0 + np.exp(-x))


# def softmax(x, axis=-1):
#     x -= np.max(x, axis=axis, keepdims=True)
#     if x.dtype == np.float32 or x.dtype == np.float64:
#         np.exp(x, out=x)
#     else:
#         x = np.exp(x)
#     x /= np.sum(x, axis=axis, keepdims=True)
#     return x


def new_parse_mdn(
    raw: torch.Tensor, in_N: int, out_N: int, out_shape: Tuple[int, ...]
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    device = raw.device
    raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))

    n_values = (raw.shape[2] - out_N) // 2
    pred_mu = raw[:, :, :n_values]
    pred_std = torch.exp(raw[:, :, n_values : 2 * n_values])
    prob = None

    if in_N > 1:
        weights = torch.zeros(
            (raw.shape[0], in_N, out_N), device=device, dtype=raw.dtype
        )
        for i in range(out_N):
            weights[:, :, i - out_N] = torch.softmax(raw[:, :, i - out_N], dim=-1)

        if out_N == 1:
            for fidx in range(weights.shape[0]):
                idxs = torch.argsort(weights[fidx][:, 0], descending=True)
                weights[fidx] = weights[fidx][idxs]
                pred_mu[fidx] = pred_mu[fidx][idxs]
                pred_std[fidx] = pred_std[fidx][idxs]

        pred_mu_final = torch.zeros(
            (raw.shape[0], out_N, n_values), device=device, dtype=raw.dtype
        )
        pred_std_final = torch.zeros(
            (raw.shape[0], out_N, n_values), device=device, dtype=raw.dtype
        )
        for fidx in range(weights.shape[0]):
            for hidx in range(out_N):
                idxs = torch.argsort(weights[fidx, :, hidx], descending=True)
                pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
                pred_std_final[fidx, hidx] = pred_std[fidx, idxs[0]]
        prob = weights.max()
    else:
        pred_mu_final = pred_mu
        pred_std_final = pred_std

    if out_N > 1:
        final_shape = tuple([raw.shape[0], out_N] + list(out_shape))
    else:
        final_shape = tuple(
            [
                raw.shape[0],
            ]
            + list(out_shape)
        )
    final = pred_mu_final.reshape(final_shape)
    final_stds = pred_std_final.reshape(final_shape)
    return final, final_stds, prob


TOut = TypeVar("TOut")


def slice_tensor_to_dict(
    tensor: Union[np.ndarray, torch.Tensor],
    slices: Dict[str, slice],
    device="cuda",
    output_type: Type[TOut] = dict,
) -> TOut:
    result = dict()
    tensor = (
        torch.tensor(tensor, device=device)
        if not isinstance(tensor, torch.Tensor)
        else tensor
    )
    for k, slice in slices.items():
        if k not in output_type.__annotations__:
            raise ValueError(f"Key {k} not in {output_type.__annotations__}")
        result[k] = tensor[..., slice]  # type: ignore
    return result  # type: ignore


def parse_categorical_crossentropy(raw, out_shape=None):
    if out_shape is not None:
        raw = raw.reshape((raw.shape[0],) + out_shape)
    return torch.softmax(raw, dim=-1)


def parse_supercombo_outputs(
    outputs: Union[torch.Tensor, np.ndarray]
) -> SupercomboFullOutput:
    outputs = (
        torch.tensor(outputs) if not isinstance(outputs, torch.Tensor) else outputs
    )
    device = str(outputs.device)
    outputs_sliced = slice_tensor_to_dict(
        outputs,
        SUPERCOMBO_OUTPUT_SLICES,
        device=device,
        output_type=FirstSliceSupercomboOutput,
    )
    # Reshaping the outputs to their values and stds
    plan, plan_stds, plan_probs = new_parse_mdn(outputs_sliced["plan"], 5, 1, (33, 15))
    lane_lines, lane_line_stds, _ = new_parse_mdn(
        outputs_sliced["lane_lines"], 0, 0, (4, 33, 2)
    )
    road_edges, road_edge_stds, _ = new_parse_mdn(
        outputs_sliced["road_edges"], 0, 0, (2, 33, 2)
    )
    pose, pose_stds, _ = new_parse_mdn(outputs_sliced["pose"], 0, 0, (6,))
    road_transform, road_transform_stds, _ = new_parse_mdn(
        outputs_sliced["road_transform"], 0, 0, (6,)
    )
    sim_pose, sim_pose_stds, _ = new_parse_mdn(outputs_sliced["sim_pose"], 0, 0, (6,))
    wide_from_device_euler, wide_from_device_euler_stds, _ = new_parse_mdn(
        outputs_sliced["wide_from_device_euler"], 0, 0, (3,)
    )
    lead, lead_stds, _ = new_parse_mdn(outputs_sliced["lead"], 2, 3, (6, 4))
    desired_curvature, _, _ = new_parse_mdn(
        outputs_sliced["desired_curvature"], 0, 0, (1,)
    )

    # Desire and desire_pred are parsed with categorical crossentropy
    desire_state = parse_categorical_crossentropy(outputs_sliced["desire_state"], (8,))
    desire_pred = parse_categorical_crossentropy(outputs_sliced["desire_pred"], (4, 8))

    # Probability logits are converted to probabilities with sigmoid
    lane_line_probs = torch.sigmoid(outputs_sliced["lane_line_probs"])
    lead_prob = torch.sigmoid(outputs_sliced["lead_prob"])
    meta = torch.sigmoid(outputs_sliced["meta"])

    # The meta and plan tensors need to be sliced further into their components
    meta_sliced = slice_tensor_to_dict(
        meta, SUPERCOMBO_META_SLICES, output_type=MetaSliced
    )
    plan_sliced = slice_tensor_to_dict(
        plan, SUPERCOMBO_PLAN_SLICES, output_type=PlanSliced
    )
    plan_stds_sliced = slice_tensor_to_dict(
        plan_stds, SUPERCOMBO_PLAN_SLICES, output_type=PlanSliced
    )

    plan_dict: PlanFull = {
        "position": plan_sliced["position"],
        "position_stds": plan_stds_sliced["position"],
        "position_prob": plan_probs or torch.tensor(0.0),
        "velocity": plan_sliced["position"],
        "velocity_stds": plan_stds_sliced["position"],
        "acceleration": plan_sliced["acceleration"],
        "acceleration_stds": plan_stds_sliced["acceleration"],
        "t_from_current_euler": plan_sliced["t_from_current_euler"],
        "t_from_current_euler_stds": plan_stds_sliced["t_from_current_euler"],
        "orientation_rate": plan_sliced["orientation_rate"],
        "orientation_rate_stds": plan_stds_sliced["orientation_rate"],
    }
    meta_dict: MetaTensors = {
        "engaged_prob": meta_sliced["engaged"],
        "brake_disengage_probs": meta_sliced["brake_disengage"],
        "gas_disengage_probs": meta_sliced["gas_disengage"],
        "steer_override_probs": meta_sliced["steer_override"],
        "brake_3_meters_per_second_squared_probs": meta_sliced["hard_brake_3"],
        "brake_4_meters_per_second_squared_probs": meta_sliced["hard_brake_4"],
        "brake_5_meters_per_second_squared_probs": meta_sliced["hard_brake_5"],
    }

    # The pose and sim_pose tensors are sliced into their components
    pose_dict: PoseTensors = {
        "trans": pose[:, :3],
        "rot": pose[:, 3:],
        "transStd": pose_stds[:, :3],
        "rotStd": pose_stds[:, 3:],
    }
    sim_pose_dict: PoseTensors = {
        "trans": sim_pose[:, :3],
        "rot": sim_pose[:, 3:],
        "transStd": sim_pose_stds[:, :3],
        "rotStd": sim_pose_stds[:, 3:],
    }

    return {
        "plan": plan_dict,
        "lane_lines": lane_lines,
        "lane_line_probs": lane_line_probs[:, 1::2],
        "lane_line_stds": lane_line_stds[:, :, 0, 0],
        "road_edges": road_edges,
        "road_edge_stds": road_edge_stds[:, :, 0, 0],
        "lead": lead,
        "lead_stds": lead_stds,
        "lead_prob": lead_prob,
        "desire_state": desire_state,
        "meta": meta_dict,
        "desire_pred": desire_pred,
        "pose": pose_dict,
        "wide_from_device_euler": wide_from_device_euler,
        "wide_from_device_euler_std": wide_from_device_euler_stds,
        "sim_pose": sim_pose_dict,
        "road_transform": road_transform[:, :3],
        "road_transform_std": road_transform_stds[:, :3],
        "desired_curvature": desired_curvature[:, 0],
        "hidden_state": outputs_sliced["hidden_state"],
    }


def mean_l2_loss(
    pred: torch.Tensor,
    ground_truth: torch.Tensor,
):
    return (
        (((pred[:, :, :2] - ground_truth[:, :, :2]) ** 2).sum(-1) ** 0.5).mean().item()
    )


def total_loss(
    pred: Union[SupercomboFullOutput, Dict],
    ground_truth: Union[SupercomboOutputLogged, Dict],
):
    # only calculate the loss for the predicted trajectory
    # p = pred["plan"]["position"]  # (1, 33, 3)
    # gt = ground_truth["plan"]["position"]  # (1, 33, 3)
    # # l2 loss only of x and y axis
    # return float(torch.mean(torch.pow(p[:, :, :2] - gt[:, :, :2], 2)))

    def recursive_sum_count(d1, d2) -> Tuple[torch.Tensor, int]:
        total_loss = torch.tensor(0.0)
        count = 0

        for key in d1:
            if key in d2:
                lowered = key.lower()
                if "std" in lowered or "prob" in lowered or "hidden_state" in lowered:
                    continue
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    loss, num_items = recursive_sum_count(d1[key], d2[key])
                    total_loss += loss
                    count += num_items
                else:
                    total_loss += (
                        (((d1[key] - d2[key]) ** 2).sum(-1) ** 0.5).mean().item()
                    )
                    if torch.isnan(total_loss):
                        print("NAN in key", key)
                    count += 1

        return total_loss, count

    total_loss, count = recursive_sum_count(pred, ground_truth)
    if count == 0:
        return 0

    return total_loss / count


TTensorDict = TypeVar("TTensorDict", bound=Mapping)


def supercombo_tensors_at_idx(
    tensors_dict: TTensorDict, idx: int, batched=True
) -> TTensorDict:
    dic = {}
    for key, item in tensors_dict.items():
        if isinstance(item, torch.Tensor) or isinstance(item, np.ndarray):
            if batched:
                dic[key] = item[:, idx]
            else:
                dic[key] = item[idx]

        elif isinstance(item, list):
            dic[key] = item[idx]
        elif isinstance(item, dict):
            dic[key] = supercombo_tensors_at_idx(item, idx, batched=batched)
        else:
            raise ValueError("Not supported type: " + str(type(item)))
    return cast(TTensorDict, dic)


def attributeproto_fp16_to_fp32(attr):
    float32_list = np.frombuffer(attr.raw_data, dtype=np.float16)
    attr.data_type = 1
    attr.raw_data = float32_list.astype(np.float32).tobytes()


def convert_fp16_to_fp32(path):
    model = onnx.load(path)
    for i in model.graph.initializer:
        if i.data_type == 10:
            attributeproto_fp16_to_fp32(i)
    for i in itertools.chain(model.graph.input, model.graph.output):
        if i.type.tensor_type.elem_type == 10:
            i.type.tensor_type.elem_type = 1
    for i in model.graph.node:
        for a in i.attribute:
            if hasattr(a, "t"):
                if a.t.data_type == 10:
                    attributeproto_fp16_to_fp32(a.t)
    return model


def create_ort_session(path: str, fp16_to_fp32: bool):
    # os.environ["OMP_NUM_THREADS"] = "4"
    # os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    print("Onnx available providers: ", ort.get_available_providers(), file=sys.stderr)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    provider: Union[str, tuple[str, Dict[Any, Any]]]
    if (
        "OpenVINOExecutionProvider" in ort.get_available_providers()
        and "ONNXCPU" not in os.environ
    ):
        provider = "OpenVINOExecutionProvider"
    elif (
        "CUDAExecutionProvider" in ort.get_available_providers()
        and "ONNXCPU" not in os.environ
    ):
        options.intra_op_num_threads = 2
        provider = ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
    else:
        options.intra_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider = "CPUExecutionProvider"

    model_data = (
        convert_fp16_to_fp32(path).SerializeToString() if fp16_to_fp32 else path
    )
    ort_session = ort.InferenceSession(model_data, options, providers=[provider])
    print("Onnx using ", ort_session.get_providers(), file=sys.stderr)
    return ort_session
