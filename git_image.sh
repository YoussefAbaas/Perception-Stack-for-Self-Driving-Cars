#!/bin/bash

# Ask the user for login details
read -p 'Specfiy the input video path: ' inputPathVar
read -p 'Specfiy the output video path: ' outputPathVar
read -p 'Use debugging mode [1/0]: ' debugVar

python3 lanes_project.py $inputPathVar $outputPathVar $debugVar > /dev/null

