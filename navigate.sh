#! /bin/bash
comma_folder="$root_folder/carla2k19"
nohup sh ../Carla2/CARLA_0.9.15/CarlaUE4.sh &
sleep 10
poetry run python -m carla_experiments.datagen.navigate