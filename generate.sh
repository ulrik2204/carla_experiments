#! /bin/bash
# this_time = $(date)
if [ $# -eq 0 ]; then
    
    root_folder="output/$(date +"%Y-%m-%d-%H-%M-%S")"
else
    root_folder=$1
fi
comma_folder="$root_folder/carla2k19"
echo "Starting at in $root_folder"
nohup sh ../Carla2/CARLA_0.9.15/CarlaUE4.sh &
sleep 10

max_attempts=1  # 500
attempt_num=1

while true; do
    echo "Attempt $attempt_num"
    # Run your command here
    poetry run python -m carla_experiments.datagen.generate_comma2k19_data --root-folder=$comma_folder --progress-file="./progress-02-15_19-32.txt"  # TODO: Add args
    
    # Check the exit status of the command
    if [ $? -eq 0 ]; then
        echo "Command succeeded."
        break
    else
        echo "Command failed. Retrying..."
        ((attempt_num++))
        # Break the loop if the number of attempts exceeds the maximum
        if [ $attempt_num -gt $max_attempts ]; then
            echo "Maximum attempts reached. Exiting."
            break
        fi
    fi
    
    # Optional: sleep before retrying
    sleep 1
done
poetry run python -m carla_experiments.datagen.extract_comma2k19 --root-folder=$root_folder --comma2k19-folder=$comma_folder  # TODO: Add args
sleep 5
npx kill-port 2000
