def stack_the_best_models(df):
    # take the best models and stack them to see if we can get a better score
    global X_train, X_test, y_train, y_test
    global master_predictions
    global my_models
    # ^ Model
    # Stacking the top 3 models (by r2 score) to see if we can get a better score
    # Instantiate the model class
    model = my_models(df)  # instantiate the model class

    # Initialize the test train split
    X_train, X_test, y_train, y_test = model.initialize_test_train_split()

    # Initialize the pipeline
    model.initialize_pipeline("stacking")

    # Initialize the grid search
    model.initialize_gridsearch("stacking")

    # Fit the grid search
    model.fit()

    # Get the best score
    print(f"Best score: {model.get_best_score()}")
    # Get the best parameters
    print(f"Best params: {model.get_best_params()}")
    # Get the test score
    print(f"Test score: {model.get_test_score()}")
    # Get the train score
    print(f"Train score: {model.get_train_score()}")
    # Save the model
    model.save_model("stacking")
    # Save the model settings
    model.save_model_settings("stacking")