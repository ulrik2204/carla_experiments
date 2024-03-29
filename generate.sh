#! /bin/bash

# Define a function to execute when SIGINT is caught
cleanup_and_exit() {
    echo "Interrupt received. Exiting..."
    # Perform any necessary cleanup here
    npx kill-port 2000
    # Exit the script
    exit 1
}

# Trap SIGINT and call the cleanup_and_exit function when caught
trap cleanup_and_exit SIGINT

if [ $# -eq 0 ]; then
    root_folder="output/$(date +"%Y-%m-%d-%H-%M-%S")"
else
    root_folder=$1
fi
comma_folder="$root_folder/carla2k19"

max_attempts=10
attempt_num=1

while true; do
    echo "Attempt $attempt_num"
    echo "Starting in $root_folder"
    nohup sh ../Carla2/CARLA_0.9.15/CarlaUE4.sh &
    sleep 10
    poetry run python -m carla_experiments.datagen.generate_comma2k19_data --root-folder=$comma_folder --progress-file="./progress-02-15_19-32.txt"  # TODO: Add args
    
    # Check the exit status of the command
    if [ $? -eq 0 ]; then
        echo "Command succeeded."
        break
    else
        ((attempt_num++))
        if [ $attempt_num -gt $max_attempts ]; then
            echo "Maximum attempts reached. Exiting."
            break
        else
            echo "Command failed. Retrying..."
            sleep 2
            npx kill-port 2000
            sleep 2
        fi
    fi
done

poetry run python -m carla_experiments.datagen.extract_comma2k19 --root-folder=$root_folder --comma2k19-folder=$comma_folder  # TODO: Add args
sleep 2
npx kill-port 2000