# Openpilot Exploration

Some experiments with the Openpilot system, OP-Deepdive and the Carla simulator.

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

## Models trained in this thesis

In this thesis, OP-Deepdive models were trained using two optimizers: SGD and AdamW. This was because the default parameters in the OP-Deepdive repository used SGD, but in their paper, they stated they used AdamW. Both configurations used a learning rate of 10^-4 and a batch size of 48 (6 per GPU). The models trained in this thesis are:
1. OP-Deepdive trained with AdamW on Comma2k19
2. OP-Deepdive trained with SGD on Comma2k19
3. OP-Deepdive trained with AdamW on CARLA Dataset 2
4. OP-Deepdive trained with SGD on CARLA Dataset 1
5. OP-Deepdive trained with SGD on CARLA Dataset 2

Here, CARLA Dataset 1 and 2 are datasets from CARLA created in this thesis. CARLA dataset 1 is a dataset created from CARLA Town01, Town02, Town03, Town04, Town05, Town06, Town07, and Town10 by just making the CARLA autopilot drive around in these towns and collect the data as 1 minute segment videos. CARLA Dataset 2 is only sampled from Town04 and Town06 since they have a focus on highway driving, and the dataset is sampled while giving the CARLA autopilot commands to go straight whenever possible. The idea with CARLA Dataset 2 was to make it more similar to Comma2k19, but distribution analysis showed that the difference between CARLA dataset 1 and 2 are minimal. 


## Qualitative Evaluation of OP-Deepdive

The models trained on Comma2k19 and CARLA Dataset 2 were qualitatively tested using the [demo.py](https://github.com/OpenDriveLab/Openpilot-Deepdive) script from OP-Deepdive and by setting them up to drive around Town04 in CARLA.


### Video of OP-Deepdive on sample

Firstly was running infernce on the models on dataset samples and plotting their predicted trajectory. This was done using the demo.py script in the OP-Deepdive repository.

#### OP-Deepdive trained with AdamW on Comma2k19 sample

Below is a video of the qualitative performance of OP-Deepdive after training with AdamW on Comma2k19 for 100 epochs. The video shows similar results to the original reported qualitative results for OP-Deepdive, which is good. This means that this model replicates the originally reported results. 

https://github.com/ulrik2204/carla_experiments/assets/65228579/fff7ec48-7d45-415a-960f-48d87496d67e



#### OP-Deepdive trained with SGD on Comma2k19 sample

Below is a video of an OP-Deepdive model trained on Comma2k19 with SGD for 100 epochs. The video shows that the model's performance is not good and predicts the same trajectory for each frame, only changing its predicted speed. This shows considerably worse performance than the originally reported results. 

https://github.com/ulrik2204/carla_experiments/assets/65228579/62206995-8b5b-4ea4-a038-a3822c74969d



#### OP-Deepdive trained with AdamW on CARLA sample

The OP-Deepdive model was also trained on a CARLA dataset with both the AdamW and SGD optimizers. Below is the plotted trajectory of the model trained with AdamW with the otherwise same configuration on CARLA Dataset 2. This obtains respectable results. 

https://github.com/ulrik2204/carla_experiments/assets/65228579/25b42b3f-401f-4b8f-99ba-a093c688c5fc



#### OP-Deepdive trained with SGD on CARLA sample

Below is the plotted trajectory on the same sample when the OP-Deepdive model is trained with SGD instead with the otherwise same hyperparameters and training setup. This model, similar to the other model trained on SGD, it gets bad results. It does respond somewhat to bends, but not enough. 


https://github.com/ulrik2204/carla_experiments/assets/65228579/cecd2767-fc15-4fe4-8c3e-def0c2b62744




### OP-Deepdive in CARLA

The models trained on Comma2k19 and on CARLA Dataset 2 were also qualitatively evalauted by being set up to drive around Town04 in CARLA. The goal with this was to see if they were able to drive at all. 

#### OP-Deepdive trained with AdamW on Comma2k19 in CARLA

Below is the video of the OP-Deepdive model trained with AdamW on Comma2k19 being set up in CARLA. The vehicle drives and is able to handle a shallow turn on the highway, although it crosses the lane lines on some occasions. One can conclude that the model gets acceptable results when set up to drive in CARLA. (The video is of low-qualityy to be able to upload it to GitHub)

https://github.com/ulrik2204/carla_experiments/assets/65228579/bc893b45-634a-4820-a3ff-babcaee5b2db


#### OP-Deepdive trained with SGD on Comma2k19 in CARLA

Below is a video of the OP-Deepdive model trained on Comma2k19 being deployed in the CARLA simulator on Town04. As we can see in the performance on the Comma2k19 sample, it predicts the same trajectory regradless of input, and as a result crashes into the guard rails soon after starting to drive. 

https://github.com/ulrik2204/carla_experiments/assets/65228579/c01837e2-b390-493c-8df8-d307fef653ea


#### OP-Deepdive trained with AdamW on CARLA Dataset 2 in CARLA

This is the video of the OP-Deepdive model trained with AdamW on CARLA Dataset 2. This model is able to follow the line and handle the shallow highway bend, but predicts a much slower speed than the model trained on Comma2k19. This is likely because of the difference in speed distributions between the datasets.

https://github.com/ulrik2204/carla_experiments/assets/65228579/dea597ba-2702-4260-b252-e8d0159dde7c



#### OP-Deepdive trained with SGD on CARLA Dataset 2 in CARLA

This is the video of the OP-Deepdive model trained with SGD on CARLA Dataset 2. This shows that the model predicts a similar driving pattern regardless of input, and as a result it crashes quite quickly.


https://github.com/ulrik2204/carla_experiments/assets/65228579/dd34feff-5a17-4899-ac1d-6d9ee7504ac6





## Video of Supercombo

The following are videos of the Supercombo model. The videos include both the original outputs of Supercombo from the Openpilot dataset, and outputs of Supercombo being actively run on the dataset using the provided OpenpilotDataset class in this repo. These videos prove that one can reproduce the outputs of Supercomobo by running inference on the Supercombo model, and that the provided OpenpilotDataset class is sufficient to extract the necessary data to run the Supercombo model. There is a small discrepancy between the outputs of the original Supercombo model and the actively run model. This can be explained by the fact that the Openpilot dataset does not have the previous hidden states of the Supercombo model. Since Supercombo is a recurrent neural network, it needs to "warm up". This is the reason why the actively run model slowly becomes more similar to the original output, as the hidden states also become more similar. 

### Predicted trajectory

Below is a video of the predicted plan of the Supercombo model. It shows the original output during data generation (called "original_plan") in red, and the actively run model (called "pred_plan"). The confidence value for "pred_plan" is also shown along with the mean L2 distance to the original output.

https://github.com/ulrik2204/carla_experiments/assets/65228579/3633e905-d017-4392-8b62-96bfb369ffdd




### Lane Lines

Below is a video of the predicted lane lines of the Supercombo model. Its shows the original predicted lane lines during data generation ("gt_lane_line") in red and the actively run model (called "pred_lane_line") along with the confidence values for "pred_lane_line" and the mean L2 distance to the original output. The line is colored darker the less confident the prediction is and is thus black for very unlikely predicted lane lines. 


https://github.com/ulrik2204/carla_experiments/assets/65228579/8af1e213-f0bd-4df7-afdc-2a7777e56801





### Road edges

Below is a video of the Supercombo model's predicted road edges. It shows the original predicted road edges during data generation ("gt_road_edge") in red and the actively run model (called "pred_road_edge") are shown, along with the confidence values for "pred_road_edge" and the mean L2 distance to the original output. The line is colored darker the less confident the prediction is and is thus black for very unlikely predicted road edges. 


https://github.com/ulrik2204/carla_experiments/assets/65228579/be4498c4-470b-4741-aae6-13923d82c1a9





### Leading vehicle

Below is a video of the Supercombo model's predicted leading vehicle(s). It shows both the original predicted leading vehicle during data generation ("gt_lead") in red and the actively run model (called "pred_lead"), along with the confidence values for "pred_lead" and the mean L2 distance to the original output. The box for the leading vehicle is colored darker the less confident the prediction is and is thus black for very unlikely predicted leading vehicles. 


https://github.com/ulrik2204/carla_experiments/assets/65228579/5b200ef3-43d3-4e20-8dad-f81ee8a30929







