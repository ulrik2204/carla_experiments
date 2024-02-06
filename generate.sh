#! /bin/bash
# this_time = $(date)
root_folder="output/$(date +"%Y-%m-%d-%H-%M-%S")"
comma_folder="$root_folder/carla2k19"
echo "Starting at in $root_folder"
nohup sh ../Carla2/CARLA_0.9.15/CarlaUE4.sh &
sleep 10
poetry run python -m carla_experiments.datagen.generate_comma2k19_data --root-folder=$comma_folder  # TODO: Add args
poetry run python -m carla_experiments.datagen.extract_comma2k19 --root-folder=$root_folder --comma2k19-folder=$comma_folder  # TODO: Add args
sleep 5
npx kill-port 2000
