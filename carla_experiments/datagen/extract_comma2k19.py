from __future__ import print_function

import glob
import random
from pathlib import Path

import click
import cv2

random.seed(0)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def _extract_sequence_id(sequence_path: str):
    return sequence_path.split("/")[-1]


# From https://github.com/OpenDriveLab/Openpilot-Deepdive/blob/main/tools/extract_comma2k19.py
@click.command()
@click.option("--root-folder", type=str, default="data")
@click.option("--comma2k19-folder", type=str, default="data/comma2k19")
def main(root_folder: str, comma2k19_folder: str):
    # root_path = Path(root_folder)
    # comma2k19_path = Path(comma2k19_folder)
    sequences = glob.glob(comma2k19_folder + "/*/*/*/video.hevc")
    random.shuffle(sequences)

    num_seqs = len(sequences)
    print(num_seqs, "sequences")

    num_train = int(0.8 * num_seqs)
    print("num_train", num_train)
    print("num_seqs", num_seqs)

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
    # frame_times = np.load(example_segment_folder + "/global_pose/frame_times")
    # print(frame_times.shape)

    # === Generating non-overlaping seqs ===
    sequences = glob.glob(comma2k19_folder + "/*/*/*/video.hevc")
    sequences = [
        Path(seq.replace(comma2k19_folder, "").replace("/video.hevc", "")).as_posix()
        for seq in sequences
    ]
    print("seq", sequences)
    seq_names = list(set([_extract_sequence_id(seq) for seq in sequences]))
    print("seq_names", seq_names)
    num_seqs = len(seq_names)
    print("num_seqs", num_seqs)
    num_train = int(0.8 * num_seqs)
    print("num_train", num_train)
    train_seq_names = seq_names[:num_train]
    print("train_seq_names", train_seq_names)
    with open(root_folder + "/comma2k19_train_non_overlap.txt", "w") as f:
        f.writelines(
            seq + "\n"
            for seq in sequences
            if _extract_sequence_id(seq) in train_seq_names
        )
    with open(root_folder + "/comma2k19_val_non_overlap.txt", "w") as f:
        f.writelines(
            seq + "\n"
            for seq in sequences
            if _extract_sequence_id(seq) not in train_seq_names
        )


if __name__ == "__main__":
    main()
