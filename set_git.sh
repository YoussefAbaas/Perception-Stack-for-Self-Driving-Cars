#!/bin/bash

# Ask the user for login details
read -p 'Use debugging mode [1/0]: ' debugVar

python3 lanes_project.py $debugVar > /dev/null

