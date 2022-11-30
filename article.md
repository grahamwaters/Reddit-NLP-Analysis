# A Tale of Two Reddits
## NLP and ML applied to text classification
*A project by Graham Waters, 2022*

![banner](./images/1.png)

# Preface

We were approached recently by a wealthy donor who wanted to use an unusual data source for an even more unusual purpose. He stated that his consortium was performing linguistics research and needed the technical expertise of a data science mind to tend to certain facets of the problem. This client's request aligned with our interests, so we agreed immediately. [The challenge was simple](./problem_statement.md)

`Could we build a machine that can detect where a post belongs on Reddit (assuming that only two threads exist, the r/Autism, and r/OCD threads)?`

We believe that we have done that. If it had been written by Charles Dickens, and he had used the top unique words from users in these subreddits it would have started something like this:

# Chapter The First.
It was the most `Austistic` of times, it was the most `Compulsive` of times,

It was like an `unsocial` `diagnosis`, it was a `guilt`-`triggered` `germ` of a time.

it was a `son` in `school`, it was `contaminated` like a `ruminating` `floor` `ritual`

it was the epoch of `sensory`; it was the epoch of `pure` `questioning`,

it was the season of `meltdowns`; it was the season of `terrified` `washing`,

It was a day of `neurotypical` opposites; it was an `urge` to `fight` `memories`,

we were easily `awkward`; we `touched` nothing before us,

our `stressed`, `lonely` `opinions` were `honest`; our `dark`, `strange` `denials` `managed` our `peace`,

-In short, The Tale of Two Reddits is much like the tale of all of us. Yet unlike two cities bound for a bitter ending, our story is full of trails, like inroads to the future, and two threads of humanity speaking their truth under similar suns.

---

# Part 1. The Problem

---

## Data Collection

We collected posts from reddit using the Pushshift API. We used the Python library `requests` to make requests to the API and parse the JSON response.
The data we collected was a very satisfactory sample size as shown below.
* 25,750 posts from the `r/Autism` subreddit
* 41,449 posts from the `r/OCD` subreddit

We performed [feature engineering](./feature_engineering.md) on the posts to extract and create additional features that we could use to train our classification models.

The features we created are summarized below. For more details, see the [feature engineering notebook](./notebooks/feature_engineering.ipynb).

| Feature | Description |
| --- | --- |
| `title + selftext` | The title and body of the post combined. This is simply a concatenation of the two and retained the name `selftext` for consistency. |
| `selftext_length` | The length of the new selftext field. |
| `selftext_word_count` | The number of words in the selftext field. |

## [Data Cleaning and Preprocessing](./data_cleaning.md)


The two threads have highly charged, and unusual keywords that occur in one or the other and make them easily differentiable. In one sense, this represents data leakage because if our goal is purely predictive, we would want to keep those highly impactful features in the model's training data. However, because our client specifically aims to use this in a more clinical setting and extend it into the social sciences, we exclude these highly biasing terms in our analysis to make it more robust.

### Examples of Biasing Terms

| Term | Occurrences in `r/Autism` | Occurrences in `r/OCD` |
| --- | --- | --- |
| `autistic` | 1,000 | 0 |
| `ocd` | 0 | 1,000 |
<!-- update with real values -->


The results of our data cleaning and preprocessing are shown below.

| | r/Autism | r/OCD |
| --- | --- | --- |
| **Total Words** | 1,000,000 | 1,000,000 |
| **Unique Words** |  50,000 | 50,000 |
| **Total Characters** | 5,000,000 | 5,000,000 |
| **Unique Characters** |  100 | 100 |
<!-- update with real values -->
---
# Part 2. The Solution

---

## [Our Modeling](./modeling.md)

We trained a variety of models on our data and compared their performance. The results are shown below. For more details, see the [modeling notebook](./notebooks/modeling.ipynb) and the link in the title of this section.

The results of the five models are summarized below.

| Model  | Accuracy Score | Cross Validation Score | RMSE | MAE | R^2 | MSE | AUC | Thresholds |
|---------|----------------|--------------------------|------|-----|-----|-----|-------|-------------|
| Random Forest | 0.70303 | 0.70303 | 0.535 | 0.29697 | 0.0 | 0.29697 | 0.70303 | [2 1 0] |
| Logistic Regression | 0.77302 | 0.77456 | 0.47643 | 0.22698 | 0.05486 | 0.22698 | 0.73027 | [2 1 0] |
| Gradient Boosting | 0.77460 | 0.77178 | 0.47476 | 0.22540 | 0.06146 | 0.22540 | 0.73193 | [2 1 0] |
| Gaussian Naive Bayes | 0.77302 | 0.77456 | 0.47643 | 0.22698 | 0.05486 | 0.22698 | 0.73027 | [2 1 0] |
| Decision Tree | 0.77302 | 0.76642 | 0.47643 | 0.22698 | 0.05486 | 0.22698 | 0.72995 | [2 1 0] |

# Results

The **gradient boosting model performed the best**.

It had the highest accuracy score, cross validation score, AUC score, and threshold score compared to the others.

It was more accurate than the second place model by 0.00158. The gradient boosting model also had the lowest RMSE and MAE. The gradient boosting model was the best model for this specific problem application. The next best model for RMSE was the logistic regression model, which had an RMSE of 0.47643. The next best model for MAE was the decision tree model, which had an MAE of 0.22698.

The random forest model performed the worst. It had the lowest accuracy score, cross validation score, AUC score, and threshold score.

The logistic regression model performed the second best, and the decision tree model performed the third best.

The gaussian naive bayes model almost came in last place. It had the second lowest accuracy score, cross validation score, AUC score, and threshold score (only 0.0001 higher than the random forest model).

# Additional Considerations

If we applied this to the social sciences we would want to consider optimizing for a different type of metric. We recommend optimizing for the F1 score. The F1 score is a combination of the precision and recall scores. The precision score is the number of true positives divided by the number of true positives plus the number of false positives. The recall score is the number of true positives divided by the number of true positives plus the number of false negatives. The F1 score is the harmonic mean of the precision and recall scores. The F1 score is a good metric to use when we want to both avoid missing true positives (in our case avoid classifying a poster as belonging to r/OCD when they are actually a r/Autism poster) at the same time as avoiding missing an actual r/OCD poster for the opposite reason. If this was extended to a diagnostic tool, we would want to do further study on making sure this is both ethical and efficable in a pseudo-clinical setting.
