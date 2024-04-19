import itertools
import os
import sys
from pathlib import Path
from turtle import forward
from typing import Any, Dict, Mapping, Tuple, Type, TypeVar, Union, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx2pytorch import ConvertModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from carla_experiments.common.constants import (
    SUPERCOMBO_META_SLICES,
    SUPERCOMBO_OUTPUT_SLICES,
    SUPERCOMBO_PLAN_SLICES,
    SupercomboInputShapes,
)
from carla_experiments.common.openpilot_repo import download_github_file
from carla_experiments.common.types_common import (
    FirstSliceSupercomboOutput,
    MetaSliced,
    MetaTensors,
    PlanSliced,
    PlanTensors,
    PoseTensors,
    SupercomboFullNumpyInputs,
    SupercomboFullOutput,
    SupercomboFullTorchInputs,
    SupercomboPartialNumpyInput,
    SupercomboPartialOutput,
)
from carla_experiments.datasets.comma3x_dataset import Comma3xDataset, get_dict_shape

PATH_TO_ONNX = Path(".weights/supercombo.onnx")
PATH_TO_METADATA = Path(".weights/supercombo_metadata.pkl")


class SupercomboONNX:

    def __init__(self) -> None:

        # self.onnx_model = onnx.load(PATH_TO_ONNX.as_posix())
        # print("onnx_model", onnx.checker.check_model(self.onnx_model))
        self.sess = get_supercombo_onnx_model(PATH_TO_ONNX)
        # TODO: need to get CUDAExecutionProvider working

    def __repr__(self) -> str:
        return str(self.sess)

    def __call__(self, inputs: SupercomboFullNumpyInputs) -> np.ndarray:
        pred = self.sess.run(None, inputs)
        return pred[0]  # only use the first result which is the [1, 6504] tensor


class SupercomboTorch(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        onx = onnx.load(PATH_TO_ONNX.as_posix())
        self.model = ConvertModel(onx)

    def forward(self, x: SupercomboFullTorchInputs):
        return self.model(x)


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = raw.device
    raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))

    n_values = (raw.shape[2] - out_N) // 2
    pred_mu = raw[:, :, :n_values]
    pred_std = torch.exp(raw[:, :, n_values : 2 * n_values])

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
    return final, final_stds


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
    for k, shape in slices.items():
        if k not in output_type.__annotations__:
            raise ValueError(f"Key {k} not in {output_type.__annotations__}")
        result[k] = tensor[:, shape]  # type: ignore
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
    print("outputs", outputs.shape)
    # TODO: Can't just put 0 here, that removed the batch size!
    device = str(outputs.device)
    outputs_sliced = slice_tensor_to_dict(
        outputs,
        SUPERCOMBO_OUTPUT_SLICES,
        device=device,
        output_type=FirstSliceSupercomboOutput,
    )
    # Reshaping the outputs to their values and stds
    plan, plan_stds = new_parse_mdn(outputs_sliced["plan"], 5, 1, (33, 15))
    lane_lines, lane_line_stds = new_parse_mdn(
        outputs_sliced["lane_lines"], 0, 0, (4, 33, 2)
    )
    road_edges, road_edge_stds = new_parse_mdn(
        outputs_sliced["road_edges"], 0, 0, (2, 33, 2)
    )
    pose, pose_stds = new_parse_mdn(outputs_sliced["pose"], 0, 0, (6,))
    road_transform, road_transform_stds = new_parse_mdn(
        outputs_sliced["road_transform"], 0, 0, (6,)
    )
    sim_pose, sim_pose_stds = new_parse_mdn(outputs_sliced["sim_pose"], 0, 0, (6,))
    wide_from_device_euler, wide_from_device_euler_stds = new_parse_mdn(
        outputs_sliced["wide_from_device_euler"], 0, 0, (3,)
    )
    lead, lead_stds = new_parse_mdn(outputs_sliced["lead"], 2, 3, (6, 4))
    desired_curvature, _ = new_parse_mdn(
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
        plan[:, 0], SUPERCOMBO_PLAN_SLICES, output_type=PlanSliced
    )
    plan_stds_sliced = slice_tensor_to_dict(
        plan_stds[:, 0], SUPERCOMBO_PLAN_SLICES, output_type=PlanSliced
    )

    plan_dict: PlanTensors = {
        "position": plan_sliced["position"],
        "position_stds": plan_stds_sliced["position"],
        "velocity": plan_sliced["position"],
        "acceleration": plan_sliced["acceleration"],
        "t_from_current_euler": plan_sliced["t_from_current_euler"],
        "orientation_rate": plan_sliced["orientation_rate"],
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
    print("pose", pose.shape)
    print("pose_stds", pose_stds.shape)
    pose_dict: PoseTensors = {
        "trans": pose[:, :3],
        "rot": pose[:, 3:],
        "transStd": pose_stds[:, :3],
        "rotStd": pose_stds[:, 3:],
    }
    print("sim_pose", sim_pose.shape)
    print("sim_pose_stds", sim_pose_stds.shape)
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
        # TODO: Continue end slicing here
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


def total_loss(pred: dict, ground_truth: dict) -> float:
    diffs = []
    for key, value in pred.items():
        if isinstance(value, torch.Tensor):
            diff = float(torch.sum(value - ground_truth[key]))
            diffs.append(diff)
        if isinstance(value, dict):
            diff = float(total_loss(value, ground_truth[key]))
            diffs.append(diff)
    return sum(diffs) / len(diffs)


TTensorDict = TypeVar("TTensorDict", bound=Mapping)


def supercombo_tensors_at_idx(tensors_dict: TTensorDict, idx) -> TTensorDict:
    dic = {}
    for key, item in tensors_dict.items():
        if isinstance(item, torch.Tensor) or isinstance(item, np.ndarray):
            dic[key] = item[:, idx]
        elif isinstance(item, dict):
            dic[key] = supercombo_tensors_at_idx(item, idx)
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
    return model.SerializeToString()


def create_ort_session(path, fp16_to_fp32):
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

    model_data = convert_fp16_to_fp32(path) if fp16_to_fp32 else path
    ort_session = ort.InferenceSession(model_data, options, providers=[provider])
    print("Onnx using ", ort_session.get_providers(), file=sys.stderr)
    return ort_session


def main():
    torch_model = SupercomboTorch()
    print(torch_model)

    return
    model = SupercomboONNX()
    segment_start_idx = 300
    segment_end_idx = 400
    batch_size = 1

    segment_length = segment_end_idx - segment_start_idx
    dataset = Comma3xDataset(
        folder="/home/ulrikro/datasets/CommaAI/2024_02_28_Orkdal",
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        device="cpu",
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in tqdm(dataloader, position=1):
        inputs, ground_truth = batch
        inputs_np = cast(
            SupercomboPartialNumpyInput,
            torch_dict_to_numpy(inputs, dtype=np.float32),
        )
        # ground_truth_np = torch_dict_to_numpy(ground_truth, dtype=np.float32)

        # Recurrent inputs
        features_buffer = np.zeros(
            (batch_size,) + SupercomboInputShapes.FEATURES_BUFFER, dtype=np.float32
        )
        prev_desired_curv = np.zeros(
            (batch_size,) + SupercomboInputShapes.PREV_DESIRED_CURV, dtype=np.float32
        )
        desire = np.zeros(
            (batch_size,) + SupercomboInputShapes.DESIRES, dtype=np.float32
        )
        for i in tqdm(range(segment_length), position=2):
            partial_inputs = supercombo_tensors_at_idx(inputs_np, i)
            inputs: SupercomboFullNumpyInputs = {
                "big_input_imgs": partial_inputs["big_input_imgs"],
                "input_imgs": partial_inputs["input_imgs"],
                "traffic_convention": partial_inputs["traffic_convention"],
                "lateral_control_params": partial_inputs["lateral_control_params"],
                "desire": desire,
                "prev_desired_curv": prev_desired_curv,
                "features_buffer": features_buffer,
            }
            gt = supercombo_tensors_at_idx(ground_truth, i)
            print(get_dict_shape(inputs))
            pred = model(inputs)
            parsed_pred = parse_supercombo_outputs(pred)
            print("out sizes", get_dict_shape(parsed_pred))
            print("\nground_truth sizes\n", get_dict_shape(ground_truth))
            compare_pred = cast(
                SupercomboPartialOutput,
                {
                    key: value
                    for key, value in parsed_pred.items()
                    if key != "hidden_state"
                },
            )
            loss = total_loss(compare_pred, gt)  # type: ignore
            # print("pred", pred)
            print("loss", loss)
            return


if __name__ == "__main__":
    main()
