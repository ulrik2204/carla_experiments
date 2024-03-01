#! /bin/bash

cleanup_and_exit() {
    echo "Interrupt received. Exiting..."
    # Perform any necessary cleanup here
    npx kill-port 2000
    # Exit the script
    exit 1
}

# Trap SIGINT and call the cleanup_and_exit function when caught
trap cleanup_and_exit SIGINT

nohup sh ../Carla2/CARLA_0.9.15/CarlaUE4.sh &
sleep 10
poetry run python -m carla_experiments.carla_eval.eval_op_deepdive
# Check the exit status of the command
if [ $? -eq 0 ]; then
    echo "Command succeeded."
    npx kill-port 2000
    exit 0
else
    echo "Command failed."
    sleep 2
    npx kill-port 2000
    exit 1
fi