# Running_Evaluation

## Introduction
This repository serves to detect and analyze a person's running form. At the moment functionality is provided for the user to evaluate their stride distance (the distance their feet travel with every running step).

This repository utilizes the PeekingDuck framework developed by AI Singapore (https://github.com/aisingapore/PeekingDuck). 

## Installation
Create a conda environment using the provided 'conda.yml' file through the command line (Note that development and testing was performed in the Windows 10 Operating System):
```
conda env create -f conda.yml
```
Activate the environment
```
conda activate pkd_running
```

## Data Input
Place a video file (preferably in .mp4 format) of a running sequence taken from the side (saggital plane) in the `data/input` folder. Ensure that almost all of the runner's body is visible in the sequence from the feet. It is recommended that the video is recorded in high quality to prevent blurring. 

## Configuration
In the `pipeline_config.yml` file, edit the `source` variable to ensure the file path, reflects the releative path for the file with your video

In the event your video is too dark, it's possible to augment it's brightness through the node `augment.brightness:`, by adjusting the `beta` value. For further information on the existing nodes and other nodes refer to https://peekingduck.readthedocs.io/en/stable/ . 

Note the path of the output data (`data/output/keypoints.csv` by default). This is where the csv file containing the pose information over tile will be stored.

## Execution 

With the configs properly set and the environment activated in the command line simply execute the following: 
```
peekingduck run
```
A pop-up window will display the video with poses imposed on the runner (if the code was successful), as well as the stride distance (distance between the left and right feet) on the top left corner. 
## Evaluation
For the purpose of exploring the data numerically, the `data_analysis.ipynb` notebook has been provided. 