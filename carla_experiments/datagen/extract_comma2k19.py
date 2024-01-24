from __future__ import print_function

import glob
import random

import click
import cv2
import numpy as np

random.seed(0)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# From https://github.com/OpenDriveLab/Openpilot-Deepdive/blob/main/tools/extract_comma2k19.py
@click.command()
@click.option("--root-folder", type=str, default="data")
@click.option("--comma2k19-folder", type=str, default="data/comma2k19")
@click.option(
    "--example-segment-folder",
    type=str,
    default="data/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/3",
)
def main(root_folder: str, comma2k19_folder: str, example_segment_folder: str):
    # root_path = Path(root_folder)
    # comma2k19_path = Path(comma2k19_folder)
    sequences = glob.glob(comma2k19_folder + "/*/*/*/video.hevc")
    random.shuffle(sequences)

    num_seqs = len(sequences)
    print(num_seqs, "sequences")

    num_train = int(0.8 * num_seqs)

    with open(root_folder + "/comma2k19_train.txt", "w") as f:
        f.writelines(
            seq.replace(comma2k19_folder, "").replace("/video.hevc", "\n")
            for seq in sequences[:num_train]
        )
    with open(root_folder + "/comma2k19_val.txt", "w") as f:
        f.writelines(
            seq.replace(comma2k19_folder, "").replace("/video.hevc", "\n")
            for seq in sequences[num_train:]
        )
    frame_times = np.load(example_segment_folder + "/global_pose/frame_times")
    print(frame_times.shape)

    # === Generating non-overlaping seqs ===
    sequences = glob.glob(comma2k19_folder + "/*/*/*/video.hevc")
    sequences = [
        seq.replace(comma2k19_folder, "").replace("/video.hevc", "")
        for seq in sequences
    ]
    seq_names = list(set([seq.split("/")[1] for seq in sequences]))
    num_seqs = len(seq_names)
    num_train = int(0.8 * num_seqs)
    train_seq_names = seq_names[:num_train]
    with open(root_folder + "/comma2k19_train_non_overlap.txt", "w") as f:
        f.writelines(
            seq + "\n" for seq in sequences if seq.split("/")[1] in train_seq_names
        )
    with open(root_folder + "/comma2k19_val_non_overlap.txt", "w") as f:
        f.writelines(
            seq + "\n" for seq in sequences if seq.split("/")[1] not in train_seq_names
        )


if __name__ == "__main__":
    main()
