# Welcome to the Main File for Project 3

# File Explanations
# main.py - This file is the main file for the project. It is the file that is run to start the program.
# data_exploration.py - This file is for exploring the data and creating plots to visualize in report.ipynb
# feature_engineering.py - This script is for feature engineering and creating new features for the model. It is not intended to be run as a script, but rather imported into the model script.
# model.py - This file is for creating the model and running the model.
# report.ipynb - This file is for creating the report for the project.
# word_lists.py - This file contains the lists of medications and conditions that are used in the feature_engineering.py script.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from the scripts
from feature_engineering import run_feature_engineering
from preprocessing import run_preprocess_data
from modeling import run_modeling # This is the function that runs the models

# Logging Setup
import logging
# set the logfile to be 'logs/main.log'
logging.basicConfig(filename='logs/main.log')


#& Step One. Import the data
# Read in the data
df = pd.read_csv("data/reddit_threads.csv") # The combined data
print(f'Completed Step One. The shape of the data is {df.shape}')

#& Step Two. Preprocess the data
df_preprocessed = run_preprocess_data(df)
print(f'Completed Step Two. The shape of the data is {df_preprocessed.shape}')

#& Step Three. Run the feature engineering functions on the data
# Run the feature engineering script
# selftext is converted to string now.
df = run_feature_engineering(df_preprocessed)
print(f'Completed Step Three. The shape of the data is {df.shape}')

#& Step Four. Run the modeling script to test the models
# Run the modeling script
run_modeling(df)
print(f'Completed Step Four. The shape of the data is {df.shape}, and we generated our model results.')