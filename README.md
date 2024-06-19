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


## Video of OP-Deepdive on sample

Below is a video of an OP-Deepdive model trained on Comma2k19 for 99 epochs evaluated qualitatively on a sample from the Comma2k19 dataset it was trained on. As seen in the video, the model's performance is not good and predicts the same trajectory for each frame, only changing its predicted speed. Despite showing the same data that was used during training, the performance is still bad. This is contrary to what was reported for OP-Deepdive. 

https://github.com/ulrik2204/carla_experiments/assets/65228579/b1c38b9c-ba50-4b75-ae36-52e88658f2d4


A version of the OP-Deepdive model trained on CARLA data from Town04 and Town06 in the same format as the Comma2k19 dataset was also evaluated qualitatively on a sample from the trained data. This vidoe shows the same story as the one above, but as the CARLA dataset was a bit more biased towards turning left, the model is constantly predicting a slightly left-turning trajectory. In the video it is clear that the model predicts a different long-term trajectory during bends than when driving straight. This shows that OP-Deepdive trained on CARLA data does attempt to handle bends as opposed to the model trained on Comma2k19. Despite this, the predicted trajectory for bends is only slight and noisy, making the performance still bad. 


https://github.com/ulrik2204/carla_experiments/assets/65228579/0d6ee0de-de1e-4405-b188-9a0a13ba4526




## Video if OP-Deepdive in CARLA

Below is a video of the OP-Deepdive model trained on Comma2k19 being deployed in the CARLA simulator on Town04. As we can see in the performance on the Comma2k19 sample, it predicts the same trajectory regradless of input, and as a result crashes into the guard rails soon after starting to drive. 

https://github.com/ulrik2204/carla_experiments/assets/65228579/c01837e2-b390-493c-8df8-d307fef653ea


A version of the OP-Deepdive model trained on CARLA data from Town04 and Town06 in the same format as the Comma2k19 dataset was also set up to run in CARLA. This shows that the model predicts a similar driving pattern when trained on CARLA data as well.


https://github.com/ulrik2204/carla_experiments/assets/65228579/dd34feff-5a17-4899-ac1d-6d9ee7504ac6





## Video of Supercombo on the Openpilot dataset





https://github.com/ulrik2204/carla_experiments/assets/65228579/84c28227-8966-496d-ac0a-2f8b9f083be0








