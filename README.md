# Carla Experiments

Some experiments with the Carla simulator.

## Setup
This project uses python 3.8.10 because it is currently the latest version of python supported by CARLA. I suggest using pyenv to ensure you are using the correct version of python. This project also uses poetry to manage dependencies.

```bash
# First time setup
poetry shell # Activate the virtual environment
poetry install
```

## PyTorch
This project uses PyTorch for its ML models. As the PyTorch library can be different depending on your system, you have to install it yourself. See the [PyTorch website](https://pytorch.org/get-started/locally/) for more information. Generally, install it with the following command (and add the correct --index-url if needed)

```bash
poetry run python -m pip install torch torchvision
```


## Recreating the Comma 2k19 dataset

Data sources to recreate in CARLA from the Comma 2k19 dataset:
- CAN messages (includes: RADAR, steering angle, wheel speed)
- Road-facing camera at 20Hz
- Raw GNSS
- IMU (accelerometer and gyroscope data)


## Video of OP-Deepdive on Comma2k19 sample

Below is a video of an OP-Deepdive model trained on Comma2k19 for 99 epochs evaluated qualitatively on a sample from the Comma2k19 dataset. As seen in the video, the model's performance is not good and predicts the same trajectory for each frame, only changing its predicted speed. This is contrary to what was reported for OP-Deepdive. 
![Video](./new_comma2k19_1_on_comma.mp4)

