from pathlib import Path
from typing import cast

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from carla_experiments.common.constants import SupercomboInputShapes
from carla_experiments.common.types_common import (
    SupercomboFullNumpyInputs,
    SupercomboPartialNumpyInput,
    SupercomboPartialOutput,
)
from carla_experiments.datasets.comma3x_dataset import Comma3xDataset, get_dict_shape
from carla_experiments.models.supercombo_utils import (
    get_supercombo_onnx_model,
    parse_supercombo_outputs,
    supercombo_tensors_at_idx,
    torch_dict_to_numpy,
    total_loss,
)

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


def main_onnx():
    print("Using ONNX")
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
        inputs_base, ground_truth = batch
        inputs_np = cast(
            SupercomboPartialNumpyInput,
            torch_dict_to_numpy(inputs_base, dtype=np.float32),
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
    main_onnx()
