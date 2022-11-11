import pandas as pd
# This script is for preprocessing the data.
import os

# Logging Setup
import logging
# set the logfile to be 'logs/preprocessing.log'
logging.basicConfig(filename='logs/preprocessing.log')


from word_lists import biasing_terms, stop, listofknown_medications
# stop words from nltk


#^ Functions

def process_dataframe(df):

    # Quick Eliminations
    if "selftext" in df.columns:
        df = df[
            ["title", "selftext", "subreddit", "author", "created_utc"]
        ]  # Eliminate the Unused Columns
    else:  # selftext has already been renamed to selftext
        df = df[["title", "selftext", "subreddit", "author", "created_utc"]]
    # Rename 'selftext' to 'selftext'
    df = df.rename(columns={"selftext": "selftext"})

    # Eliminate the Duplicates
    df = df.drop_duplicates()

    # Eliminate the Null Values
    df = df.dropna()

    # There are some rows that have a Selftext of '[removed]' or '[deleted]' which are not useful
    # We want to drop these rows.
    df = df[df["selftext"] != "[removed]"]
    df = df[df["selftext"] != "[deleted]"]

    # We want to drop the rows that have a Selftext of ' ' or '  ' or '   ' etc.
    df = df[df["selftext"] != " "]

    # We also want to drop any rows that have an author that has been deleted or removed.
    df = df[df["author"] != "[deleted]"]
    df = df[df["author"] != "[removed]"]

    # Lowercase the selftext and title columns
    df["selftext"] = df["selftext"].apply(lambda x: x.lower())
    df["title"] = df["title"].apply(lambda x: x.lower())

    # Remove the punctuation from the selftext and title columns
    df["selftext"] = df["selftext"].str.replace("[^\w\s]", "")
    df["title"] = df["title"].str.replace("[^\w\s]", "")

    # Remove the numbers from the selftext and title columns
    df["selftext"] = df["selftext"].str.replace("\d+", "")
    df["title"] = df["title"].str.replace("\d+", "")

    # Remove the stopwords from the selftext and title columns
    df["selftext"] = df["selftext"].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop)
    )
    df["title"] = df["title"].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop)
    )

    # Remove the common words from the selftext and title columns
    freq = pd.Series(" ".join(df["selftext"]).split()).value_counts()[:10]
    freq = list(freq.index)
    df["selftext"] = df["selftext"].apply(
        lambda x: " ".join(x for x in x.split() if x not in freq)
    )
    freq = pd.Series(" ".join(df["title"]).split()).value_counts()[:10]
    freq = list(freq.index)
    df["title"] = df["title"].apply(
        lambda x: " ".join(x for x in x.split() if x not in freq)
    )
    # source: https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/


    # Remove np.nan values from the selftext column (include any missing values in this removal)
    df = df[df["selftext"].notna()]

    return df

def remove_overly_biasing_terms(df):
    # adapted from source: https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
    # remove the overly biasing terms from the selftext in the dataframe
    df["selftext"] = df["selftext"].apply(
        lambda x: " ".join(x for x in x.split() if x not in biasing_terms)
    )
    return df

def merge_text(df):
    """
    Merges text in the title and selftext columns into the selftext column. This is done to increase the amount of text in the selftext column and eliminate the issues that could come up with title-heavy posts.

    Args:
        df (dataframe): The dataframe to be processed.

    Raises:
        Exception: If the dataframe does not have a title and selftext column.

    Returns:
        df: The processed dataframe.
    """
    # # 0. Rename selftext to selftext #^ note: used to be post_text but replaced 39 times with selftext on 2022-10-08
    # Merge the title and selftext columns into one column
    if "title" in df.columns:
        df["selftext"] = df["title"] + " " + df["selftext"]
    else:
        raise Exception("Step 3. in preprocessing > merge_text() is not needed because the title column is not in the dataframe.")
    return df

def preprocessing_function(df):
    """
    Run the process_dataframe and merge_text functions on the dataframe.

    Args:
        df (dataframe): The dataframe to be processed.

    Returns:
        dataframe: The processed dataframe.
    """
    df = merge_text(df)  # introduce new features
    df = process_dataframe(df)  # process the dataframe
    df = remove_overly_biasing_terms(
        df
    )  # remove the overly biasing terms from the selftext in the dataframe
    return df

def data_exploration(df):
    """
    Perform data exploration on the dataframe.

    Args:
        df (dataframe): The dataframe to be explored.
    """
    # We want to explore the data to see if there is anything we can do to improve the model.
    # We want to see if there is any correlation between the length of the title and the length of the selftext
    # and the subreddit that the post is from.
    sns.scatterplot(x="title_length", y="selftext_length", hue="is_autism", data=df)
    plt.show()

    # We want to see if there is any correlation between the length of the title and the subreddit that the post is from.
    sns.scatterplot(x="title_length", y="is_autism", data=df)
    plt.show()

    # We want to see if there is any correlation between the length of the selftext and the subreddit that the post is from.
    sns.scatterplot(x="selftext_length", y="is_autism", data=df)
    plt.show()

    # We want to see if there is any correlation between the length of the title and the length of the selftext.
    sns.scatterplot(x="title_length", y="selftext_length", data=df)
    plt.show()

    # We want to see if there is any correlation between the length of the title and the subreddit that the post is from.
    sns.scatterplot(x="title_length", y="is_autism", data=df)
    plt.show()


def run_preprocess_data(df):

    # ^ File I/O
    # If the df_cleaned.csv file has not been generated yet, then we want to generate it with the preprocessing_function.
    # If the df_cleaned.csv file has been generated, then we want to read it in and use it for the model.

    if not os.path.isfile("data/df_cleaned.csv"):
        logging.info("The df_cleaned.csv file has not been generated yet, so we are generating it now.")
        # Create df by concatenating the dataframes from the two subreddits
        #& Reading in both Reddit threads as csv files.
        df1 = pd.read_csv("./data/ocd_thread.csv")  # read in the ocd threads
        df2 = pd.read_csv("./data/autism_thread.csv")  # read in the autism threads
        df = pd.concat([df1, df2], ignore_index=True)  # combine the two dataframes
    else:
        logging.info(f'File "data/df_cleaned.csv" exists. Reading it in now.')
        # Read the dataframe from the df_cleaned.csv file
        df = pd.read_csv("data/df_cleaned.csv")

    logging.info(f'Preprocessing the dataframe...')
    #& Preprocessing the data in the dataframe.
    df = preprocessing_function(df)  # Preprocess the Dataframe

    # Save the Processed Dataframe to a CSV
    logging.info(f'Saving the processed dataframe to "data/df_cleaned.csv"...')
    df.to_csv("./data/df_cleaned.csv")

    return df # return the dataframe