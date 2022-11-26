# This file is for exploring the data and creating plots to visualize in report.ipynb

# This data is from two threads on Reddit, the r/Autism, and r/OCD thread. The data was collected using the pushshift.io API.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Logging Setup
import logging

# set the logfile to be 'logs/data_exploration.log'
logging.basicConfig(filename="logs/data_exploration.log")

# Import functions from feature_engineering.py
from feature_engineering import feature_engineer, remove_ocd_meds

# Read in the data
df = pd.read_csv("data/reddit_threads.csv")  # The combined data

# # Explore the data
# 1. Top Bigrams in the post selftext for r/Autism
# 2. Top Bigrams in the post selftext for r/OCD
# 3. Top Trigrams in the post selftext for r/Autism
# 4. Top Trigrams in the post selftext for r/OCD
# 5. Top Words in the post selftext for r/Autism
# 6. Top Words in the post selftext for r/OCD
# 7. Top Users posting on r/Autism in the data
# 8. Top Users posting on r/OCD in the data
# 9. The top words used by the top users in r/Autism using groupby
# 10. The top words used by the top users in r/OCD using groupby

# Questions to Ask:
# 1. Are there any outliers in the data?
# 2. Are there any words that are used more often in one subreddit than the other?
# 3. Are there any words that are used more often by one user than the other?
# 4. Are there any words that are used more often by one user in one subreddit than the other?
# 5. What are areas for potential data leakage?

# Distributions
# Plot the distribution(s) of the following features:
# 1. The number of words in the post selftext
# 2. The number of characters in the post selftext
# 3. The number of sentences in the post selftext
# 4. The posts per user in r/Autism
# 5. The posts per user in r/OCD
# 6. The number of words per user in r/Autism
# 7. The number of words per user in r/OCD
# 8. The number of characters per user in r/Autism
# 9. The number of characters per user in r/OCD


# Visualization Functions
# List of visualizations to create:
# * 1. Bar plot of the number of posts per subreddit, subplotted by autism and ocd.

# * 2. Top Twenty Five words in each subreddit, subplotted by autism and ocd.
