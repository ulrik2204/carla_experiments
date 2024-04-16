import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Type, TypedDict, TypeVar, Union, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
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
    MetaTensors,
    PlanTensors,
    SupercomboOutput,
)
from carla_experiments.datasets.comma3x_dataset import Comma3xDataset, get_dict_shape

PATH_TO_ONNX = Path(".weights/supercombo.onnx")
PATH_TO_METADATA = Path(".weights/supercombo_metadata.pkl")


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


class SupercomboInputs(TypedDict):
    input_imgs: np.ndarray
    big_input_imgs: np.ndarray
    desire: np.ndarray
    traffic_convention: np.ndarray
    lateral_control_params: np.ndarray
    prev_desired_curv: np.ndarray
    features_buffer: np.ndarray


class Supercombo:

    def __init__(self, mode: Literal["infer"] = "infer") -> None:

        if not PATH_TO_ONNX.exists():
            download_github_file(
                repo_owner="commaai",
                repo_name="openpilot",
                file_path_in_repo="selfdrive/modeld/models/supercombo.onnx",
                save_path=PATH_TO_ONNX,
                saved_at="git-lfs",
                main_branch_name="master",
            )
        # self.onnx_model = onnx.load(PATH_TO_ONNX.as_posix())
        # print("onnx_model", onnx.checker.check_model(self.onnx_model))
        print("available", ort.get_available_providers())
        print("pat", PATH_TO_ONNX.as_posix())
        self.sess = create_ort_session(PATH_TO_ONNX.as_posix(), fp16_to_fp32=True)
        # TODO: need to get CUDAExecutionProvider working
        ins = [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in self.sess.get_inputs()
        ]
        outputs = [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in self.sess.get_outputs()
        ]
        print("inputs", ins)
        print("outputs", outputs)

    def __repr__(self) -> str:
        return str(self.sess)

    def __call__(self, inputs: SupercomboInputs) -> np.ndarray:
        pred = self.sess.run(None, inputs)
        return pred


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x -= np.max(x, axis=axis, keepdims=True)
    if x.dtype == np.float32 or x.dtype == np.float64:
        np.exp(x, out=x)
    else:
        x = np.exp(x)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


def new_parse_mdn(input_tensor, in_N=0, out_N=1, out_shape=None):
    raw = input_tensor
    raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))
    # print("raw1", raw.shape, raw)

    n_values = (raw.shape[2] - out_N) // 2
    pred_mu = raw[:, :, :n_values]
    pred_std = np.exp(raw[:, :, n_values : 2 * n_values])
    # print("pred_mu1", pred_mu.shape, pred_mu)

    if in_N > 1:
        weights = np.zeros((raw.shape[0], in_N, out_N), dtype=raw.dtype)
        for i in range(out_N):
            # print("weights index shape", raw[:,:,i - out_N].shape)
            weights[:, :, i - out_N] = softmax(raw[:, :, i - out_N], axis=-1)

        if out_N == 1:
            for fidx in range(weights.shape[0]):
                idxs = np.argsort(weights[fidx][:, 0])[::-1]
                weights[fidx] = weights[fidx][idxs]
                pred_mu[fidx] = pred_mu[fidx][idxs]
                pred_std[fidx] = pred_std[fidx][idxs]

        pred_mu_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
        pred_std_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
        for fidx in range(weights.shape[0]):
            for hidx in range(out_N):
                idxs = np.argsort(weights[fidx, :, hidx])[::-1]
                pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
                pred_std_final[fidx, hidx] = pred_std[fidx, idxs[0]]
        # print("pred_mu2", pred_mu_final.shape, pred_mu_final)
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


# T = TypeVar("T", bound=Union[np.ndarray, torch.Tensor])
TOut = TypeVar("TOut")


def slice_tensor_to_dict(
    tensor: Union[np.ndarray, torch.Tensor],
    slices: Dict[str, slice],
    _: Type[TOut] = dict,
) -> TOut:
    result = dict()
    for k, shape in slices.items():
        sliced = tensor[shape]  # type: ignore
        result[k] = (
            np.copy(sliced) if type(sliced) is np.ndarray else torch.clone(sliced)  # type: ignore
        )
    return result  # type: ignore


def parse_categorical_crossentropy(raw, out_shape=None):
    if out_shape is not None:
        raw = raw.reshape((raw.shape[0],) + out_shape)
    return softmax(raw, axis=-1)


def parse_supercombo_outputs(outputs: np.ndarray) -> SupercomboOutput:
    outputs_sliced = slice_tensor_to_dict(outputs, SUPERCOMBO_OUTPUT_SLICES)

    meta_sliced = slice_tensor_to_dict(
        outputs_sliced["meta"], SUPERCOMBO_META_SLICES, MetaTensors
    )
    plan_sliced = slice_tensor_to_dict(
        outputs_sliced["plan"], SUPERCOMBO_PLAN_SLICES, PlanTensors
    )

    plan: PlanTensors = {
        "position": 0,
        "position_stds": 0,
        "velocity": 0,
        "acceleration": 0,
        "t_from_current_euler": 0,
        "orientation_rate": 0,
    }

    return {
        "plan": torch.Tensor(),
        "lane_lines": torch.Tensor(),
        "lane_line_probs": torch.Tensor(),
        "lane_line_stds": torch.Tensor(),
        "road_edges": torch.Tensor(),
        "lead": torch.Tensor(),
        "lead_stds": torch.Tensor(),
        "lead_prob": torch.Tensor(),
        "desire_state": torch.Tensor(),
        "meta": torch.Tensor(),
        "desire_pred": torch.Tensor(),
        "pose": torch.Tensor(),
        "wide_from_device_euler": torch.Tensor(),
        "wide_from_device_euler_std": torch.Tensor(),
        "sim_pose": torch.Tensor(),
        "road_transform": torch.Tensor(),
        "road_transform_std": torch.Tensor(),
        "desired_curvature": torch.Tensor(),
        "hidden_state": torch.Tensor(),
    }


def main():

    model = Supercombo()
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
        inputs_np = torch_dict_to_numpy(inputs, dtype=np.float32)
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
            inputs: SupercomboInputs = {
                "input_imgs": inputs_np["input_imgs"][:, i, :, :, :].transpose(
                    0, 3, 1, 2
                ),
                "big_input_imgs": inputs_np["big_input_imgs"][:, i, :, :, :].transpose(
                    0, 3, 1, 2
                ),
                "traffic_convention": inputs_np["traffic_convention"][:, i, :],
                "lateral_control_params": inputs_np["lateral_control_params"][:, i, :],
                "desire": desire,
                "prev_desired_curv": prev_desired_curv,
                "features_buffer": features_buffer,
            }
            print(get_dict_shape(inputs))
            pred = model(inputs)
            print("pred", pred)
            return


if __name__ == "__main__":
    main()
