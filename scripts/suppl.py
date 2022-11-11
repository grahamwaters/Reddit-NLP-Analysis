
### start of paste


model_names = {
    "logreg": LogisticRegression(),
    "dt": DecisionTreeClassifier(),
    "adaboost": AdaBoostClassifier(),
    "rf": RandomForestClassifier()
}

### Count Vectorizer
# Instantiate the CountVectorizer
vectorizer = CountVectorizer()
print(df["selftext"].isnull().sum())

corpus = df["selftext"]
# source: https://stackoverflow.com/a/39308809/12801757 for below
cvec = vectorizer.fit(corpus)  # fit and transform the data to the self-text column

##! section one. Nonlemmatized fields

X = df["selftext"]  # post
y = df["is_autism"]  # predicting if the post is on the autism subreddit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


def model_iterator(model, pipe_params, model_names, lemmatized_bool):
    """
    Run a model with a set of parameters

    Args:
        model (str): alias and name of the model
        pipe_params (dict): _description_
        lemmatized_bool (bool): whether the model is lemmatized or not

    Returns:
        model: _description_
    """
    name_model = model_names[model]  # get the model from the dictionary
    # Creating Pipeline
    pipe = Pipeline([("cvec", CountVectorizer()), (f"{str(model)}", model)])
    # Instantiate GridSearchCV.
    gs_model = GridSearchCV(
        pipe,  # what object are we optimizing?
        param_grid=pipe_params,  # what parameters values are we searching?
        cv=3,
        n_jobs=-1,
    )  # 3-fold cross-validation. source: https://stackoverflow.com/questions/24121018/sklearn-gridsearch-how-to-print-out-progress-during-the-execution
    if lemmatized_bool:
        filename = "../models/lemmatized" + str(model) + "_model.pkl"
    else:
        filename = "../models/" + str(model) + "_model.pkl"
    pickle.dump(gs_model, open(filename, "wb+"))
    return gs_model  # return the model


lemmatizedBoolean = False

# 1. Logistic Regression
# * Original Params
# pipe_params_sent_len = {
#     'cvec__max_features': [1000, 2000, 3000],
#     'cvec__min_df': [2, 3],
#     'cvec__max_df': [.9, .95],
#     'cvec__ngram_range': [(1,1), (1,2)],
#     'logreg__penalty': ['l1','l2'],
#     'logreg__C': [1, 2, 3]
# }

# ? Final Params
pipe_params = {
    "cvec__max_features": [3000],
    "cvec__min_df": [2],
    "cvec__max_df": [0.9],
    "cvec__ngram_range": [(1, 1)],
    "logreg__penalty": ["l2"],
    "logreg__C": [1, 2, 3],
}
logregmodel = model_iterator("logreg", pipe_params, model_names, lemmatizedBoolean)


# 2. Decision Tree
pipe_params_sent_len = {
    "cvec__max_features": [1000, 2000, 3000],
    "cvec__min_df": [2, 3],
    "cvec__max_df": [0.9, 0.95],
    "cvec__ngram_range": [(1, 1), (1, 2)],
    "dt__max_depth": [None, 2, 3, 4],
    "dt__min_samples_split": [2, 3, 4],
    "dt__min_samples_leaf": [1, 2, 3],
}

# Running the model
dt_model = model_iterator("dt", pipe_params_sent_len, model_names, lemmatizedBoolean)


# 3. AdaBoost Model
pipe_params_sent_len = {
    "cvec__max_features": [1000, 2000, 3000],
    "cvec__min_df": [2, 3],
    "cvec__max_df": [0.9, 0.95],
    "cvec__ngram_range": [(1, 1), (1, 2)],
    "ada__n_estimators": [50, 100, 150],
    "ada__learning_rate": [0.1, 0.5, 1],
}

# Running the model
ada_model = model_iterator("ada", pipe_params_sent_len, model_names, lemmatizedBoolean)

#! section two. Lemmatized fields
lemmatizedBoolean = True

X = df["selftext_lemmatized"]  # post
y = df["is_autism"]  # predicting if the post is on the autism subreddit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 1. Logistic Regression

# params
ipe_params = {
    "cvec__max_features": [3000],
    "cvec__min_df": [2],
    "cvec__max_df": [0.9],
    "cvec__ngram_range": [(1, 1)],
    "logreg__penalty": ["l2"],
    "logreg__C": [1, 2, 3],
}

logregmodel_lemmatized = model_iterator(
    "logreg", pipe_params, model_names, lemmatizedBoolean
)


# 2. Decision Tree
pipe_params_sent_len = {
    "cvec__max_features": [1000, 2000, 3000],
    "cvec__min_df": [2, 3],
    "cvec__max_df": [0.9, 0.95],
    "cvec__ngram_range": [(1, 1), (1, 2)],
    "dt__max_depth": [None, 2, 3, 4],
    "dt__min_samples_split": [2, 3, 4],
    "dt__min_samples_leaf": [1, 2, 3],
}

# Running the model
dt_model_lemmatized = model_iterator(
    "dt", pipe_params_sent_len, model_names, lemmatizedBoolean
)

# 3. AdaBoost Model
pipe_params_sent_len = {
    "cvec__max_features": [1000, 2000, 3000],
    "cvec__min_df": [2, 3],
    "cvec__max_df": [0.9, 0.95],
    "cvec__ngram_range": [(1, 1), (1, 2)],
    "ada__n_estimators": [50, 100, 150],
    "ada__learning_rate": [0.1, 0.5, 1],
}

# Running the model
ada_model_lemmatized = model_iterator(
    "ada", pipe_params_sent_len, model_names, lemmatizedBoolean
)

#! Fitting Models (Takes Time)
# fit all the models on the training data
print(f"Fitting Models on unlemmatized data")
logregmodel.fit(X_train, y_train)
print("Logistic Regression Model Fitted")
dt_model.fit(X_train, y_train)
print("Decision Tree Model Fitted")
ada_model.fit(X_train, y_train)
print("AdaBoost Model Fitted")

print(f"Fitting Models on lemmatized data")
logregmodel_lemmatized.fit(X_train, y_train)
print("Logistic Regression Model Fitted")
dt_model_lemmatized.fit(X_train, y_train)
print("Decision Tree Model Fitted")
ada_model_lemmatized.fit(X_train, y_train)
print("AdaBoost Model Fitted")

allmodels = [
    logregmodel,
    dt_model,
    ada_model,
    logregmodel_lemmatized,
    dt_model_lemmatized,
    ada_model_lemmatized,
]  # list of all models

#! Results Display
for every_model in allmodels:
    print("Results for: ", every_model)
    print(f"    Training score: {every_model.score(X_train, y_train)}")
    print(f"    Testing score: {every_model.score(X_test, y_test)}")
    print(f"    Best score: {every_model.best_score_}")
    print(f"    Best params: {every_model.best_params_}")
    print(f"    Best estimator: {every_model.best_estimator_}")



self.X = (df["selftext"],)
        self.y = df["is_autism"]  #
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, random_state=self.random_state
        )



    def model_operation(self, df):
        # note: be sure that the selftext col has no null values before passing it to the model function.

        # Instantiate the CountVectorizer
        vectorizer = CountVectorizer()
        df["selftext"] = df["selftext"].fillna(
            ""
        )  # fill in the NaN values with empty strings

        corpus = df["selftext"]  # create a corpus of the selftext column

        # source: https://stackoverflow.com/a/39308809/12801757 for below
        cvec = vectorizer.fit(
            corpus
        )  # fit and transform the data to the self-text column

        X = df["selftext"]  # post
        y = df["is_autism"]  # predicting if the post is on the autism subreddit

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # ignore warnings

        # try adaboost with logistic regression as the base estimator
        warnings.filterwarnings("ignore")
        pipe_sent_len = Pipeline(
            [("cvec", CountVectorizer()), ("dt", DecisionTreeClassifier())]
        )

        pipe_params_sent_len = {
            "cvec__max_features": [1000, 2000, 3000],
            "cvec__min_df": [2, 3],
            "cvec__max_df": [0.9, 0.95],
            "cvec__ngram_range": [(1, 1), (1, 2)],
            "dt__max_depth": [None, 2, 3, 4],
            "dt__min_samples_split": [2, 3, 4],
            "dt__min_samples_leaf": [1, 2, 3],
        }

        # Instantiate GridSearchCV.
        gs_dt = GridSearchCV(
            pipe_sent_len,  # what object are we optimizing?
            param_grid=pipe_params_sent_len,  # what parameters values are we searching?
            cv=3,
            n_jobs=-1,
        )  # , verbose = 10) # 3-fold cross-validation. source: https://stackoverflow.com/questions/24121018/sklearn-gridsearch-how-to-print-out-progress-during-the-execution

        gs_dt.fit(X_train, y_train)

        return gs_dt  # return the model which contains the best parameters for the model, and the best score.



### end of paste
        self.cvec = CountVectorizer()
        self.logreg = LogisticRegression()
        self.decisiontree = DecisionTreeClassifier()
        self.randomforest = RandomForestClassifier()
        self.adaboost = AdaBoostClassifier()
        self.gradientboosting = GradientBoostingClassifier()
        self.rf = RandomForestClassifier()
        self.ada = AdaBoostClassifier()
        self.gbc = GradientBoostingClassifier()
        self.xgb = XGBClassifier()