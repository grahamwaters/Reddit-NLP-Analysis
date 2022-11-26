import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import tqdm
from tqdm import tqdm
import os
import warnings

# Logging Setup
import logging

# set the logfile to be 'logs/modeling.log'
logging.basicConfig(filename="logs/modeling.log")

import json

# import countvectorizer
from sklearn.feature_extraction.text import CountVectorizer

# import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, GridSearchCV, and AdaBoostClassifier.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import train_test_split

from word_lists import stop  # stop words

# suppress future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Different Models to use for Classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

# import plot_roc_curve
from sklearn.metrics import plot_roc_curve

# Standard Scaler and Pipeline Imports

from sklearn.pipeline import Pipeline

# import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# import XGBClassifier
from xgboost import XGBClassifier

# overly biasing terms:
from word_lists import biasing_terms
from modeling import stop


class my_models:
    def __init__(self, df):
        # params grid for logistic regression.
        self.params_logreg = {
            "cvec__max_features": [3000, 4000, 5000],
            "cvec__min_df": [2, 3, 4],
            "cvec__max_df": [0.9, 1, 0.1],
            "cvec__ngram_range": [(1, 1)],
            "logreg__penalty": ["l1", "l2"],
            "logreg__C": [1, 2, 3],
        }
        self.params_decisiontree = {
            "cvec__max_features": [3000],
            "cvec__min_df": [2],
            "cvec__max_df": [0.9],
            "cvec__ngram_range": [(1, 1)],
            "decisiontree__max_depth": [2, 3, 4],
            "decisiontree__min_samples_split": [2, 3, 4],
            "decisiontree__min_samples_leaf": [1, 2, 3],
        }
        self.params_randomforest = {
            "cvec__max_features": [3000],
            "cvec__min_df": [2],
            "cvec__max_df": [0.9],
            "cvec__ngram_range": [(1, 1)],
            "randomforest__n_estimators": [50, 100, 150],
            "randomforest__max_depth": [2, 3, 4],
            "randomforest__min_samples_split": [2, 3, 4],
            "randomforest__min_samples_leaf": [1, 2, 3],
        }
        self.params_adaboost = {
            "cvec__max_features": [1000, 2000, 3000],
            "cvec__min_df": [2, 3, 4],
            "cvec__max_df": [0.9],
            "cvec__ngram_range": [(1, 1)],
            "adaboost__n_estimators": [50, 100, 150, 200],
            "adaboost__learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        self.params_gradientboosting = {
            "cvec__max_features": [1000, 2000, 3000],
            "cvec__min_df": [2],
            "cvec__max_df": [0.9],
            "cvec__ngram_range": [(1, 1)],
            "gradientboosting__n_estimators": [50, 100, 150],
            "gradientboosting__max_depth": [2, 3, 4],
            "gradientboosting__min_samples_split": [2, 3, 4],
            "gradientboosting__min_samples_leaf": [1, 2, 3],
        }
        self.params_xgboost = (
            {
                "cvec__max_features": [3000],
                "cvec__min_df": [2],
                "cvec__max_df": [0.9],
                "cvec__ngram_range": [(1, 1)],
                "xgboost__n_estimators": [100],
                "xgboost__learning_rate": [0.1],
                "xgboost__max_depth": [3],
                "xgboost__min_child_weight": [1],
                "xgboost__gamma": [0],
                "xgboost__subsample": [1],
                "xgboost__colsample_bytree": [1],
                "xgboost__reg_alpha": [0],
                "xgboost__reg_lambda": [1],
            },
        )

        # X_train, X_test, y_train, y_test
        self.X_train = []  # initialize,
        self.X_test = []  # initialize,
        self.y_train = []  # initialize,
        self.y_test = []  # initialize,
        self.random_state = 42
        self.df = df

    # methods
    def get_params(self, model):
        # retrieves the parameters for the model.
        if model == "logreg":
            return self.params_logreg
        elif model == "decisiontree":
            return self.params_decisiontree
        elif model == "randomforest":
            return self.params_randomforest
        elif model == "adaboost":
            return self.params_adaboost
        elif model == "gradientboosting":
            return self.params_gradientboosting
        elif model == "xgboost":
            return self.params_xgboost
        else:
            return None

    def initialize_test_train_split(self):
        df = self.df
        X = df["selftext"]  # features
        y = df["is_autism"]  # target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=self.random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def initialize_pipeline(self, model):
        """
        Initialize the pipeline.

        Args:
            model (str): model to use in the pipeline.
        """
        if model == "logreg":
            self.pipe = Pipeline(
                [("cvec", CountVectorizer()), ("logreg", LogisticRegression())]
            )
        elif model == "decisiontree":
            self.pipe = Pipeline(
                [
                    ("cvec", CountVectorizer()),
                    ("decisiontree", DecisionTreeClassifier()),
                ]
            )
        elif model == "randomforest":
            self.pipe = Pipeline(
                [
                    ("cvec", CountVectorizer()),
                    ("randomforest", RandomForestClassifier()),
                ]
            )
        elif model == "adaboost":
            self.pipe = Pipeline(
                [("cvec", CountVectorizer()), ("adaboost", AdaBoostClassifier())]
            )
        elif model == "gradientboosting":
            self.pipe = Pipeline(
                [
                    ("cvec", CountVectorizer()),
                    ("gradientboosting", GradientBoostingClassifier()),
                ]
            )
        elif model == "xgboost":
            self.pipe = Pipeline(
                [("cvec", CountVectorizer()), ("xgboost", XGBClassifier())]
            )
        else:
            self.pipe = None

    # Methods for GridSearchCV.
    def initialize_gridsearch(self, model):
        """
        Initialize the grid search.

        Args:
            model (str): model to use in the grid search.
        """
        self.gs = GridSearchCV(
            self.pipe, param_grid=self.get_params(model), cv=5, n_jobs=-1
        )

    def fit(self):
        """
        Fit the grid search.
        """
        self.gs.fit(self.X_train, self.y_train)

    def get_best_score(self):
        """
        Get the best score from the grid search.
        """
        return self.gs.best_score_

    def get_best_params(self):
        """
        Get the best parameters from the grid search.
        """
        return self.gs.best_params_

    def get_test_score(self):
        """
        Get the test score from the grid search.
        """
        return self.gs.score(self.X_test, self.y_test)

    def get_train_score(self):

        """
        Get the train score from the grid search.
        """
        return self.gs.score(self.X_train, self.y_train)

    def run_gridsearch(self, model):

        """
        Run the grid search.

        Args:
            model (str): model to use in the grid search.
        """
        self.initialize_pipeline(model)
        self.initialize_gridsearch(model)
        self.fit()  # fit the grid search
        logging.info(f"Best score: {self.get_best_score()}")
        logging.info(f"Best params: {self.get_best_params()}")
        logging.info(f"Test score: {self.get_test_score()}")
        logging.info(f"Train score: {self.get_train_score()}")
        return self.gs

    def save_model(self, model):
        """
        Save the model.

        Args:
            model (str): model to save.
        """
        pickle.dump(self.gs, open(f"../models/{model}.pkl", "wb"))

    def load_model(self, model):
        """
        Load the model.

        Args:
            model (str): model to load.
        """
        self.gs = pickle.load(open(f"../models/{model}.pkl", "rb"))

    def save_model_settings(self, model):
        """
        Save the model settings.

        Args:
            model (str): model to save.
        """
        with open(f"../model_settings/{model}.json", "w") as f:
            json.dump(self.get_best_params(), f)


def run_modeling():
    """
    Problem Statement:
    A wealthy donor with a track record of philanthropic contributions to both Autism and OCD research organizations contacted our organization, asking for a model that they can utilize to identify post characteristics on Reddit.
    The purposes of this study (towards those ends) are to:

    1) Use Pushshift API to scrape Reddit posts from the Autism and OCD subreddits.
    2) To build a predictive model that can accurately predict whether a post is from the Autism or OCD subreddit

    To accomplish these goals, we hypothesize that count vectorization, and Logistic Regression, Adaboost, or Decision Trees can be used to build a model that accurately can predict whether a post is from the Autism or OCD subreddit. Success in this study would mean that our model has a misclassification rate of less than 10 percent and an accuracy score of greater than 90 percent on the test data set.
    """

    # Control Flow:
    # We already performed the data collection and saved the data to csv files in the data folder.
    # We will now read the data from the csv files and perform data exploration and preprocessing.

    # ^ Model
    # map 1 to autism and 0 to ocd based on subreddit column, then drop the subreddit column
    df["is_autism"] = df["subreddit"].map({"autism": 1, "OCD": 0})
    df.drop(columns=["subreddit"], inplace=True)

    # ^ Modeling and Evaluation
    # Instantiate the model class
    model = my_models(df)  # instantiate the model class

    # Initialize the test train split
    X_train, X_test, y_train, y_test = model.initialize_test_train_split()

    # For each model in the list of models, initialize the pipeline, grid search, fit the grid search, and print the best score, best parameters, test score, and train score, and append the best score to the list of best scores.
    # also, save a plot of the ROC curve to the images folder, and append all predictions to the master_predictions df in a new column for each model.

    models = [
        "logreg",
        "decisiontree",
        "randomforest",
        "adaboost",
        "gradientboosting",
        "xgboost",
    ]
    best_scores = []
    master_predictions = pd.DataFrame()

    for model_name in tqdm(models):
        model.run_gridsearch(
            model_name
        )  # include the model binary classification as arg
        # save the model as a pickled file in the models folder, also save all parameters and settings used to create the model as a json file in the models_settings folder
        model.save_model(model_name)
        # save settings for the model as a json file in the models_settings folder.
        model.save_model_settings(model_name)
        best_scores.append(model.get_best_score())
        master_predictions[f"{model_name}_predictions"] = model.gs.predict(X_test)

    # ^ Model Evaluation (Macro Average)
    # Create a dataframe of the best scores for each model
    best_scores_df = pd.DataFrame(best_scores, index=models, columns=["best_score"])

    # Save the best scores df to a csv file
    best_scores_df.to_csv("./model_results/best_scores.csv")

    # Plot all model scores as a bar chart with the y-axis as the model score on the testing data, and the x-axis as the model name.
    best_scores_df.plot(kind="bar")
    plt.title("Model Scores")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.ylabel("Model Score")
    plt.savefig("images/model_scores.png")


# & Run the modeling script by uncommenting the line below
# run_modeling()

# & Works Cited:
# https://stackoverflow.com/questions/24121018/sklearn-gridsearch-how-to-print-out-progress-during-the-execution
# preprocessing_functions and data_exploration functions were adapted from the NLP lesson in the GA Data Science course.
# source: https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
