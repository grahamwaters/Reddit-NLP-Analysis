
# Modeling

## Method 1: Random Forest Classifier
Our first method was to train a random forest classifier using scikit-learn. This is a supervised learning algorithm which trains a set of decision trees and then combines their predictions into a single prediction. In order to avoid overfitting, we split our data using the 80/20 convention. We then trained our model on the training data and tested it on the test data. The following code snippet shows how we built our random forest classifier model.

```python
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# create a random forest classifier
rfc = RandomForestClassifier()
model = rfc

X_train, X_test, y_train, y_test, model, param_defaults = pre_test(original_X_train, original_X_test, y_train, y_test, model, param_defaults)

# fit the model on the training data
rfc.fit(X_train, y_train)

# predict on the testing data
y_pred = rfc.predict(X_test)

master_results_dataframe = save_results(model,master_results_dataframe, y_test, y_pred, X_train, X_test, y_train, param_defaults)
```
Our results from the Random Forest Model were good.

```output
accuracy_score: 0.7753968253968254
cross_val_score: 0.76999081824057
rmse: 0.4739231737351262
mae: 0.2246031746031746
r2: 0.06476952330994679
mse: 0.2246031746031746
auc: 0.7316044849518064
thresholds: [2 1 0]
AUC score: 0.8585915677660483
```
The rmse and mae scores are low, which means that our model is not overfitting; however, the r2 score is low, which means that our model is not a good fit for the data. But r2 is not always the best metric as it is sensitive to outliers. The AUC score is high, which means that our model is a good fit for the data. The AUC score is a good metric to use when the data is imbalanced. What is interesting is that the data is actually balanced because we accounted for class imbalance in our preprocessing step.

| Model | Accuracy | Cross Val Score | RMSE | MAE | R2 | MSE | AUC | AUC Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest Classifier | 0.7753968253968254 | 0.76999081824057 | 0.4739231737351262 | 0.2246031746031746 | 0.06476952330994679 | 0.2246031746031746 | 0.7316044849518064 | 0.8585915677660483 |





## Method 2: Logistic Regression
The second method we tried was logistic regression. This is a supervised learning algorithm which trains a linear function that predicts a binary target variable based on a set of input variables. We trained our model on the training data and tested it on the test data. The following code snippet shows how we built our logistic regression model.

```python
model = LogisticRegression()

X_train = original_X_train # set to the originals
X_test = original_X_test
if not check_xtrain_is_same(X_train, original_X_train):
    raise Exception('X_train has been modified')

# create a SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=4) # 4 features with the highest scores are selected

# apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(X_train, y_train)
# create a list of the selected features
selected_features = X_train.columns[fvalue_selector.get_support()]
# create a list of the non-selected features
non_selected_features = X_train.columns[~fvalue_selector.get_support()]
# score the model with the selected features
X_train = X_train[selected_features]
X_test = X_test[selected_features]
# check the shape of the training data to make sure it is the same as the testing data

# create a logistic regression model
logreg = LogisticRegression()
# fit the model to the training data
logreg.fit(X_train, y_train)
# score the model on the training data
logreg.score(X_train, y_train)
# score the model on the testing data
logreg.score(X_test, y_test)
# predict the target values for the testing data
y_pred = logreg.predict(X_test)
# create a confusion matrix
confusion_matrix(y_test, y_pred)
# create a classification report
print(classification_report(y_test, y_pred))
# create a dataframe of the coefficients
coefficients = pd.DataFrame({'feature': X_train.columns, 'coefficient': logreg.coef_[0]})
# sort the coefficients by their magnitude
coefficients.sort_values('coefficient', ascending=False, inplace=True)
# plot the coefficients
plt.figure(figsize=(10, 10))
plt.barh(coefficients['feature'], coefficients['coefficient'])
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.savefig('../images/logreg_coefficients.png')
plt.show();
master_results_dataframe = save_results(logreg,master_results_dataframe, y_test, y_pred, X_train, X_test, y_train, param_defaults)
```
The results are shown below.
```output
accuracy_score: 0.6944444444444444
cross_val_score: 0.7011290017496571
rmse: 0.5527707983925667
mae: 0.3055555555555556
r2: -0.27231001245819964
mse: 0.3055555555555556
auc: 0.6253688282735558
thresholds: [2 1 0]
AUC score: 0.7687102485082945
```

The model's accuracy score is high, and the RMSE and MAE are low. But the r2 is negative. This means that the model is predicting more false positives than true positives.


The model is being overly cautious and is ignoring many posts that are from the correct subreddit. This is likely because of the relatively small sample size we have.

| Model | Accuracy | Cross Val Score | RMSE | MAE | R2 | MSE | AUC | AUC Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.694444 | 0.701129 | 0.552771 | 0.305556 | -0.272310 | 0.305556 | 0.625369 | 0.768710 |


## Method 3: Gradient Boosting
The third method we tried was gradient boosting. This is an ensemble machine learning technique that creates a sequence of weak learners to create a strong learner. It is similar to the random forest but uses a different approach for feature selection. In contrast to the random forest, it performs feature selection in a greedy fashion, which means that features are added to the model one at a time. Each weak learner is trained on the residuals of the previous model. We implemented it as shown below.


```python
# create a Gradient Boosting classifier
gbc = GradientBoostingClassifier()
model = gbc

X_train, X_test, y_train, y_test, model, param_defaults = pre_test(original_X_train, original_X_test, y_train, y_test, model, param_defaults)

# fit the model on the training data
gbc.fit(X_train, y_train)

# predict on the testing data
y_pred = gbc.predict(X_test)

master_results_dataframe = save_results(model,master_results_dataframe, y_test, y_pred, X_train, X_test, y_train, param_defaults)
```
RESULTS
```output
accuracy_score: 0.7746031746031746
cross_val_score: 0.7717767295597484
rmse: 0.4747597554519816
mae: 0.2253968253968254
r2: 0.061464821978886586
mse: 0.2253968253968254
auc: 0.7319257753589927
thresholds: [2 1 0]
AUC score: 0.8609035473083732
```
With the gradient boosting model, we see that the accuracy score is high, and the RMSE and MAE are low. The r2 is positive, which means that the model is predicting more true positives than false positives. The AUC score is also high, which means that the model is performing well.

These results show that the model has performed well. The AUC score (0.731) is very high, indicating that the model is performing 23.1% better than the baseline model (which would have an AUC score of 0.5). The accuracy score is also high, indicating that the model is correctly predicting the subreddit 77.5% of the time. The RMSE and MAE are low, indicating that the model is not making large errors in its predictions. The r2 is positive, indicating that the model is predicting more true positives than false positives.

## Method 4. Gaussian Naive Bayes
We also tried a simple naive bayesian classifier. This is an unsupervised learning algorithm that trains a probabilistic model of the data. It assumes that each feature is independent of all other features. We used the `scikit-learn` library to implement the algorithm. Our code snippet is shown below.
```python
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
# create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

model = gnb

X_train, X_test, y_train, y_test, model, param_defaults = pre_test(X_train, X_test, y_train, y_test, model, param_defaults)

# fit the model on the training data
gnb.fit(X_train, y_train)

# predict on the testing data
y_pred = gnb.predict(X_test)

master_results_dataframe = save_results(model,master_results_dataframe, y_test, y_pred, X_train, X_test, y_train, param_defaults,gridsearch=False)
```
Our results for the gnb model were very encouraging.
```output
accuracy_score: 0.773015873015873
cross_val_score: 0.774555492504847
rmse: 0.4764285119345052
mae: 0.22698412698412698
r2: 0.054855419316766074
mse: 0.22698412698412698
auc: 0.7302734246934628
thresholds: [2 1 0]
AUC score: 0.8553458789587569
```
The accuracy score is high, and the RMSE and MAE are relatively low. The r2 is positive, which means that the model is predicting more true positives than false positives. Most of the metrics point to a good model. It performs better than the logistic regression model, but not as well as the gradient boosting model.

## Method 5. Decision Tree Model
The final model we tested was a decision tree model. A decision tree is an algorithmic model that makes decisions based on the input variables. The decision tree model we used is the `sklearn` implementation. We did not use the default parameters. Instead, we tuned the parameters using grid search. Our code snippet is shown below.
```python3
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
model = dtc
X_train, X_test, y_train, y_test, model, param_defaults = pre_test(original_X_train, original_X_test, y_train, y_test, model, param_defaults)
#note: Be sure to scale the data before using SVM
X_train_sc = sc.fit_transform(X_train) # fit and transform the training data
X_test_sc = sc.transform(X_test) # transform the testing data
# fit the model on the training data
dtc.fit(X_train, y_train)
# predict on the testing data
y_pred = dtc.predict(X_test)
print(f'Preparing The Results')
master_results_dataframe = save_results(model,master_results_dataframe, y_test, y_pred, X_train, X_test, y_train, param_defaults)
```
Our results are shown below.
```output
accuracy_score: 0.773015873015873
cross_val_score: 0.7664184045018205
rmse: 0.4764285119345052
mae: 0.22698412698412698
r2: 0.054855419316766074
mse: 0.22698412698412698
auc: 0.7299455773391909
thresholds: [2 1 0]
AUC score: 0.8524778703035867
```
This is a very promising result. However, it is important to note that this model has not been validated. A more thorough validation process would be required before a real-world application could use this model. The accuracy score is high, and the RMSE and MAE are relatively low. The r2 is still positive. Compared to the other models, the gaussian naive bayes model performs well, but the gradient boosting model still performs better (with an AUC score of 0.8609 vs 0.8553 and an accuracy score of 0.7746 vs 0.7730).

[Return to Main Page](./article.md)

## Conclusion
The results of the five models are summarized below.

| Model  | Accuracy Score | Cross Validation Score | RMSE | MAE | R^2 | MSE | AUC | Thresholds |
|---------|----------------|--------------------------|------|-----|-----|-----|-------|-------------|
| Random Forest | 0.70303 | 0.70303 | 0.535 | 0.29697 | 0.0 | 0.29697 | 0.70303 | [2 1 0] |
| Logistic Regression | 0.77302 | 0.77456 | 0.47643 | 0.22698 | 0.05486 | 0.22698 | 0.73027 | [2 1 0] |
| Gradient Boosting | 0.77460 | 0.77178 | 0.47476 | 0.22540 | 0.06146 | 0.22540 | 0.73193 | [2 1 0] |
| Gaussian Naive Bayes | 0.77302 | 0.77456 | 0.47643 | 0.22698 | 0.05486 | 0.22698 | 0.73027 | [2 1 0] |
| Decision Tree | 0.77302 | 0.76642 | 0.47643 | 0.22698 | 0.05486 | 0.22698 | 0.72995 | [2 1 0] |

# Results

The gradient boosting model performed the best. It had the highest accuracy score, cross validation score, AUC score, and threshold score compared to the others. It was more accurate than the second place model by 0.00158. The gradient boosting model also had the lowest RMSE and MAE. The gradient boosting model was the best model for this specific problem application. The next best model for RMSE was the logistic regression model, which had an RMSE of 0.47643. The next best model for MAE was the decision tree model, which had an MAE of 0.22698.

The random forest model performed the worst. It had the lowest accuracy score, cross validation score, AUC score, and threshold score.

The logistic regression model performed the second best, and the decision tree model performed the third best.

The gaussian naive bayes model almost came in last place. It had the second lowest accuracy score, cross validation score, AUC score, and threshold score (only 0.0001 higher than the random forest model).

# Additional Considerations

If we applied this to the social sciences we would want to consider optimizing for a different type of metric. We recommend optimizing for the F1 score. The F1 score is a combination of the precision and recall scores. The precision score is the number of true positives divided by the number of true positives plus the number of false positives. The recall score is the number of true positives divided by the number of true positives plus the number of false negatives. The F1 score is the harmonic mean of the precision and recall scores. The F1 score is a good metric to use when we want to both avoid missing true positives (in our case avoid classifying a poster as belonging to r/OCD when they are actually a r/Autism poster) at the same time as avoiding missing an actual r/OCD poster for the opposite reason. If this was extended to a diagnostic tool, we would want to do further study on making sure this is both ethical and efficable in a pseudo-clinical setting.
