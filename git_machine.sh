#!/bin/bash

# installing dependencies

# pip install moviepy
# pip install -U scikit-learn
# pip install scipy
# pip install numpy 
# pip install -U scikit-image

# Ask the user for login details
read -p 'Specfiy the input video path: ' inputPathVar
read -p 'Specfiy the output video path: ' outputPathVar

python3 car_detection_project.py $inputPathVar $outputPathVar > /dev/null

