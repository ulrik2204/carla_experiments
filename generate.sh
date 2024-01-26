#! /bin/bash
# this_time = $(date)
root_folder="output/$(date +"%Y-%m-%d-%H-%M-%S")"
echo "Starting at in $root_folder"
poetry run python -m carla_experiments.datagen.generate_comma2k19_data --root-folder=$root_folder  # TODO: Add args
# poetry run python -m carla_experiments.datagen.extract_comma2k19 --root-folder="output" --comma2k19-folder=$root_folder --example_segment_folder="$root_folder/Chunk_1/"  # TODO: Add args