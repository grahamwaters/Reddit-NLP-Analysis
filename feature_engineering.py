# This script is for feature engineering and creating new features for the model. It is not intended to be run as a script, but rather imported into the model script.

# Global Report-Level Variables
OCD_Posts_With_Meds = 0 # initialized count of posts mentioning meds from the OCD subreddit to zero.
Autism_Posts_With_Meds = 0 # initialized count of posts mentioning meds from the Autism subreddit to zero.
Total_Posts_With_Meds = 0 # the total number of posts that mention medications.
medications_mentioned = [] # the list of all medications that are mentioned in the posts.

# import the lists of medications and conditions from word_lists.py

from word_lists import listofknown_medications, conditions
import pandas as pd
import numpy as np

# Logging Setup
import logging
# set the logfile to be 'logs/feature_engineering.log'
logging.basicConfig(filename='logs/feature_engineering.log')






def remove_ocd_meds(text,listofknown_medications):
    global meds
    if len(meds)> 0:
        log_string = f'The latest medication mentioned is: {meds[-1]}'
        logging.info(log_string)
    wordsintext = text.split(' ')

    for word in wordsintext:
        if word in listofknown_medications:
            meds.append(word)
            text = text.replace(word,' ')
    # remove if any words in text are in list of known medications
    return text

#! Engineering Features from Text Data
def feature_engineer(df,listofknown_medications):
    """
    summary: This function takes in a dataframe and a list of known medications and returns a dataframe with new features.
    List of Features that will be created:
    1. Number of words in the post (word_count)
    2. Number of unique words in the post (unique_words)
    2. Length of the post (number of characters)
    4. unique words in the post (unique_word_count)
    5. If the selftext of the post contains a known medication (medication_mentioned) (list of the meds mentioned)

    Args:
        df (dataframe): The dataframe to be used for feature engineering.
        listofknown_medications (list): A list of known medications.

    Returns:
        df: A dataframe with new features.
    """
    logging.info("Engineering features...")
    # consolidate all titles and self-text into the selftext column
    df['selftext'] = df['title'] + ' ' + df['selftext'] # concatenate the title and selftext columns

    #& To engineer features here we want our selftext column to be a string, so we will convert it to a string.
    df['selftext'] = df['selftext'].astype(str)

    df["word_count"] = df["selftext"].apply(lambda x: len(str(x).split(" ")))
    df["unique_word_count"] = df["selftext"].apply(
        lambda x: len(set(str(x).split(" ")))
    )
    df["post_length"] = df["selftext"].apply(lambda x: len(str(x)))
    # create a new col in df that is boolean to indicate if any word in the selftext is in the list of known medications and is not a space or a ' or a comma
    df['medication_mentioned'] = df['selftext'].apply(lambda x: any(word in listofknown_medications for word in x.split(' ') if word not in [' ','\'',',']))
    logging.info(f'The meds mentioned column has been intialized with {df["medication_mentioned"].sum()} rows that mention medications.')






    #& Updating Global Variables for the Reporting Stage
    global medications_mentioned
    dfss = df[df['medication_mentioned'] == True] # create a new df with only the rows that mention medications
    # Update our Report-Level Variable: OCD_Posts_With_Meds to the number of posts that mention medications & are from the OCD subreddit
    global OCD_Posts_With_Meds
    OCD_Posts_With_Meds = dfss[dfss['subreddit'] == 'OCD']['medication_mentioned'].sum()
    logging.info(f'There are {OCD_Posts_With_Meds} posts that mention medications and are from the OCD subreddit.')
    # Update our Report-Level Variable: Autism_Posts_With_Meds to the number of posts that mention medications & are from the Autism subreddit
    global Autism_Posts_With_Meds
    Autism_Posts_With_Meds = dfss[dfss['subreddit'] == 'Autism']['medication_mentioned'].sum()
    logging.info(f'There are {Autism_Posts_With_Meds} posts that mention medications and are from the Autism subreddit.')
    # Update our Report-Level Variable: Total_Posts_With_Meds to the number of posts that mention medications
    global Total_Posts_With_Meds
    Total_Posts_With_Meds = dfss['medication_mentioned'].sum()
    logging.info(f'There are {Total_Posts_With_Meds} posts that mention medications.')
    # Update our Report-Level Variable: medications_mentioned to the list of medications that are mentioned in the posts. (i.e. 'Prozac', 'Zoloft', etc.)
    global medications_mentioned
    # from the dfss dataframe, create a list of all the medications mentioned in the posts as strings and append them to the medications_mentioned list.
    medications_mentioned = medications_mentioned + dfss['selftext'].apply(lambda x: [word for word in x.split(' ') if word in listofknown_medications]).tolist()
    # flatten the list of lists of medications mentioned
    medications_mentioned = [item for sublist in medications_mentioned for item in sublist]
    # remove duplicates from the list of medications mentioned
    medications_mentioned = list(set(medications_mentioned))
    logging.info(f'The medications mentioned are: {medications_mentioned}')

    return df

def binarize_target_feature(df):
    df["is_autism"] = df["subreddit"].apply(lambda x: 1 if x == "autism" else 0)
    #? Drop the 'subreddit' column
    #note: not sure if the subreddit column is needed anymore but I will keep it for now.
    return df

def run_feature_engineering(df):

    #& Importing the Data and Engineering Features
    df = pd.read_csv('./data/reddit_threads.csv')
    logging.info(f'Beginning feature engineering...')
    # Run the functions in this script to create new features for the model.
    #& binarize the target feature
    df = binarize_target_feature(df) # binarize the target feature (i.e. 'subreddit')
    #& create new features
    df = feature_engineer(df,listofknown_medications)
    # save the dataframe with the new features to a csv file 'df_after_feature_engineering.csv' in the data folder.
    #& Save those features to a csv file
    df.to_csv('./data/df_after_feature_engineering.csv',index=False)
    logging.info(f'The dataframe with the new features has been saved to the data folder as df_after_feature_engineering.csv')

    #& Save the global variables to a pandas dataframe, and then a csv named 'global_variables.csv' in the data folder.
    global_variables = pd.DataFrame({'OCD_Posts_With_Meds':[OCD_Posts_With_Meds],'Autism_Posts_With_Meds':[Autism_Posts_With_Meds],'Total_Posts_With_Meds':[Total_Posts_With_Meds],'medications_mentioned':[medications_mentioned]})
    global_variables.to_csv('./data/global_variables.csv',index=False)
    logging.info(f'The global variables have been saved to the data folder as global_variables.csv')

    return df # return the dataframe with the new features