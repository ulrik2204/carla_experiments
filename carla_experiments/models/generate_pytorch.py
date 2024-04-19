# from onnx_pytorch import code_gen
# torch.fx.wrap(len)
# torch.fx.wrap("len")
# torch.fx.wrap(torch.fx.Proxy.__len__)
# torch.fx.wrap("__len__")
import itertools
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import requests
import torch
import torch.fx
from onnx2pytorch import ConvertModel
from torch.fx import GraphModule, Tracer, symbolic_trace

PATH_TO_ONNX = Path("./carla_experiments/.weights/supercombo.onnx")
PATH_TO_METADATA = Path("./carla_experiments/.weights/supercombo_metadata.pkl")
ORT_TYPES_TO_NP_TYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(uint8)": np.uint8,
}
supercombo_metadata_dict = {
    "output_slices": {
        "plan": slice(0, 4955, None),
        "lane_lines": slice(4955, 5483, None),
        "lane_lines_prob": slice(5483, 5491, None),
        "road_edges": slice(5491, 5755, None),
        "lead": slice(5755, 5857, None),
        "lead_prob": slice(5857, 5860, None),
        "desire_state": slice(5860, 5868, None),
        "meta": slice(5868, 5916, None),
        "desire_pred": slice(5916, 5948, None),
        "pose": slice(5948, 5960, None),
        "wide_from_device_euler": slice(5960, 5966, None),
        "sim_pose": slice(5966, 5978, None),
        "road_transform": slice(5978, 5990, None),
        "desired_curvature": slice(5990, 5992, None),
        "hidden_state": slice(5992, None, None),
    },
    "input_shapes": {
        "input_imgs": (1, 12, 128, 256),
        "big_input_imgs": (1, 12, 128, 256),
        "desire": (1, 100, 8),
        "traffic_convention": (1, 2),
        "lateral_control_params": (1, 2),
        "prev_desired_curv": (1, 100, 1),
        "nav_features": (1, 256),
        "nav_instructions": (1, 150),
        "features_buffer": (1, 99, 512),
    },
    "output_shapes": {"outputs": (1, 6504)},
}


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


def create_ort_session(path, fp16_to_fp32):
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

    print("Onnx available providers: ", ort.get_available_providers(), file=sys.stderr)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    provider: str | tuple[str, dict[Any, Any]]
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
    print("Onnx selected provider: ", [provider], file=sys.stderr)
    ort_session = ort.InferenceSession(model_data, options, providers=[provider])
    print("Onnx using ", ort_session.get_providers(), file=sys.stderr)
    return ort_session


def get_model_metadata():
    with open(PATH_TO_METADATA, "rb") as f:
        model_metadata = pickle.load(f)
    # print("metadata\n", model_metadata)
    return model_metadata

    # CODE TO CREATE DUMMPY INPUT TO MODEL
    # net_output_size = model_metadata["output_shapes"]["outputs"][1]
    # output = np.zeros(net_output_size, dtype=np.float32)
    # session = create_ort_session(PATH_TO_ONNX, fp16_to_fp32=True)
    # input_names = [x.name for x in session.get_inputs()]
    # input_shapes = {x.name: [1, *x.shape[1:]] for x in session.get_inputs()}
    # input_dtypes = {x.name: ORT_TYPES_TO_NP_TYPES[x.type] for x in session.get_inputs()}
    # dummpy_input = (
    #     {k: np.zeros(input_shapes[k], dtype=input_dtypes[k]) for k in input_names},
    # )


def create_dummy_input():
    session = create_ort_session(PATH_TO_ONNX, fp16_to_fp32=True)
    input_names = [x.name for x in session.get_inputs()]
    input_shapes = {x.name: [1, *x.shape[1:]] for x in session.get_inputs()}
    input_dtypes = {x.name: ORT_TYPES_TO_NP_TYPES[x.type] for x in session.get_inputs()}
    dummpy_input = {
        k: np.zeros(input_shapes[k], dtype=input_dtypes[k]) for k in input_names
    }
    return dummpy_input


def convert_model_to_pytorch_in_memory():
    # DID NOT WORK
    # code_gen.gen("./selfdrive/modeld/models/supercombo.onnx", "./pytorch_model")
    # THIS WORKS, BUT ONLY MAKES AN IN-MEMORY MODEL, THUS NO PYTHON PYTORCH MODEL FILE
    onnx_model = onnx.load(PATH_TO_ONNX)
    pytorch_model = ConvertModel(onnx_model)
    return pytorch_model
    # symbolic_traced: GraphModule = symbolic_trace(pytorch_model)


def symbolic_conversion():
    meta = get_model_metadata()
    print("meta", meta)
    model = convert_model_to_pytorch_in_memory()
    # model("./memory_model.pth")
    model.eval()
    # example_input = create_dummy_input()
    # tracer = Tracer(autowrap_functions=(len, torch.fx.Proxy.__len__))
    # graph = tracer.trace(model)
    # name = (
    #     model.__class__.__name__
    #     if isinstance(model, torch.nn.Module)
    #     else model.__name__
    # )
    # trace = GraphModule(tracer.root, graph, name)
    # # trace = symbolic_trace(model, concrete_args=example_input)
    # print(trace.code)


def graph_onnx_model():
    onnx_model = onnx.load(PATH_TO_ONNX)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))


def main():
    # graph_onnx_model()
    if not PATH_TO_METADATA.is_file():
        download_github_file(
            repo_owner="commaai",
            repo_name="openpilot",
            file_path_in_repo="selfdrive/modeld/models/supercombo_metadata.pkl",
            save_path=PATH_TO_METADATA,
            main_branch_name="master",
        )
    if not PATH_TO_ONNX.is_file():
        download_github_file(
            repo_owner="commaai",
            repo_name="openpilot",
            file_path_in_repo="selfdrive/modeld/models/supercombo.onnx",
            save_path=PATH_TO_ONNX,
            main_branch_name="master",
        )

    symbolic_conversion()


if __name__ == "__main__":
    main()
