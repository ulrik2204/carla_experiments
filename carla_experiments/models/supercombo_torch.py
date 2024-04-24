from pathlib import Path
from typing import cast

import torch
from onnx2torch import convert
from onnx2torch.node_converters import add_converter
from torch.utils.data import DataLoader
from tqdm import tqdm

from carla_experiments.common.constants import SupercomboInputShapes
from carla_experiments.common.types_common import (
    SupercomboFullTorchInputs,
    SupercomboPartialOutput,
)
from carla_experiments.datasets.comma3x_dataset import Comma3xDataset, get_dict_shape
from carla_experiments.models.supercombo_utils import (
    convert_fp16_to_fp32,
    parse_supercombo_outputs,
    supercombo_tensors_at_idx,
    total_loss,
)

PATH_TO_ONNX = Path(".weights/supercombo.onnx")


class SupercomboTorch(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        onx = convert_fp16_to_fp32(PATH_TO_ONNX.as_posix())
        self.model = convert(onx)
        print("inputs names", self.model.input_names)
        self._convert_biases()
        self._convert_params()

    def _convert_biases(self):
        # Convert biases to float32 if not None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                if module.bias is not None:
                    module.bias.data = module.bias.data.float()

    def _convert_params(self):
        for param in self.model.parameters():
            param.data = param.data.float()

    def forward(self, x: SupercomboFullTorchInputs):
        # Convert input tensors to float16 before passing to the model
        # x_converted = {k: v.half() for k, v in x.items()}
        return self.model(**x)


def main_torch():
    print("Using torch")
    model = SupercomboTorch()
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
        inputs_base, ground_truth = batch
        # ground_truth_np = torch_dict_to_numpy(ground_truth, dtype=np.float32)

        # Recurrent inputs
        features_buffer = torch.zeros(
            (batch_size,) + SupercomboInputShapes.FEATURES_BUFFER, dtype=torch.float32
        )
        prev_desired_curv = torch.zeros(
            (batch_size,) + SupercomboInputShapes.PREV_DESIRED_CURV, dtype=torch.float32
        )
        desire = torch.zeros(
            (batch_size,) + SupercomboInputShapes.DESIRES, dtype=torch.float32
        )
        for i in tqdm(range(segment_length), position=2):
            partial_inputs = supercombo_tensors_at_idx(inputs_base, i)
            inputs: SupercomboFullTorchInputs = {
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
    main_torch()
