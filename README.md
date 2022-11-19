# OCD vs. Autism (A Reddit Thread NLP Analysis)
*A project by Graham Waters, 2022*

![banner](./images/1.png)

# Executive Summary

The end goal for our client is likely a more clinical application of classification to assist users seeking help on public forums by using psychoanalysis from text data; however, this is beyond the scope of our initial study. Instead, we hope that by learning how these subreddits present linguistically, we gain insight into the most predictive features that can serve as the first stepping stone toward such a clinical application in the future.

**Note**: This study is focused solely on linguistic features present in Reddit posts and is not a formal means of diagnosis for identifying autism-spectrum or obsessive-compulsive disorder.

---

# A Table of Contents
- [OCD vs. Autism (A Reddit Thread NLP Analysis)](#ocd-vs-autism-a-reddit-thread-nlp-analysis)
- [Executive Summary](#executive-summary)
- [A Table of Contents](#a-table-of-contents)
- [Methods](#methods)
- [About the API](#about-the-api)
- [Data Collection](#data-collection)
- [Files Provided and their Sequence](#files-provided-and-their-sequence)
- [Model Files](#model-files)
- [Data Cleaning](#data-cleaning)
  - [Balancing the Data](#balancing-the-data)
  - [User Top Word Analysis](#user-top-word-analysis)
  - [Keyword Data Leakage](#keyword-data-leakage)
  - [Steps for Data Cleaning in this Study](#steps-for-data-cleaning-in-this-study)
- [Feature Engineering](#feature-engineering)
- [Visualizing the Data](#visualizing-the-data)
- [Data Exploration](#data-exploration)
  - [Top 25 Users by Post Count in the `r/OCD` Thread](#top-25-users-by-post-count-in-the-rocd-thread)
  - [Top 25 Users by Post Count in the `r/Autism` Thread](#top-25-users-by-post-count-in-the-rautism-thread)
  - [Bigrams and Trigrams are great ways to examine text data as well.](#bigrams-and-trigrams-are-great-ways-to-examine-text-data-as-well)
    - [Model 1.1. Logistic Regression](#model-11-logistic-regression)
    - [Model 1.2. Adaboost](#model-12-adaboost)
    - [Model 1.3 Decision Tree](#model-13-decision-tree)
  - [Models using lemmatization](#models-using-lemmatization)
- [Comments on Misclassification](#comments-on-misclassification)
- [Conclusions and Recommendations](#conclusions-and-recommendations)
- [Future Work](#future-work)
- [Works Cited](#works-cited)


---

![](./images/3.jpg)


# Methods

**Problem Statement:**
A wealthy donor with a track record of philanthropic contributions to both Autism and OCD research organizations contacted our organization, asking for a model they can utilize to identify post characteristics on Reddit.
The purposes of this study (towards those ends) are to:

1) Use Pushshift API to scrape Reddit posts from the Autism and OCD subreddits.
2) To build a predictive model that can accurately predict whether a post is from the Autism or OCD subreddit

To accomplish these goals, we hypothesize that count vectorization and Logistic Regression, Adaboost, or Decision Trees can be used to build a model that accurately can predict whether a post is from the Autism or OCD subreddit. Success in this study would mean that our model has a misclassification rate of less than 10% and an accuracy score greater than 90% on the test data set.

[presentation](https://www.canva.com/design/DAFOTzanP1s/gXPNiG_2EAQ7svxm0Y6dPQ/edit?utm_content=DAFOTzanP1s&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

# About the API

Pushshift's API is relatively straightforward. For example, if We want the posts from [`/r/boardgames`](https://www.reddit.com/r/boardgames), all We have to do is use the following URL: https://api.pushshift.io/reddit/search/submission?subreddit=boardgames

# Data Collection
To gather data for this analysis, We scraped Reddit for posts on two threads, the `r/Autism` thread and the `r/OCD` thread.

Before getting too deep into the project and eliminating rows/columns from the data, We want to get a bird's eye view of what We have scraped from these subreddits.

We want the data-cleaning process to be as scalable as possible for future research, so We will use a function to clean the data. This function will clean the data for both the `Autism` and `OCD` subreddits.

# Files Provided and their Sequence
The files are ordered as follows:

1. feature_engineering.ipynb # This file contains the code for the feature engineering process.
2. data_cleaning.py # data cleaning functions that We consolidated from a data cleaning notebook for space optimization.
3. data_exploration.ipynb # The exploration of the data.
4. modeling.ipynb # the first iteration of modeling and analysis
5. modeling_beta.ipynb # A streamlined version of modeling.ipynb with added lemmatized models.

# Model Files

The models are saved as pickle files in the `models` folder. The models are named as follows:
1. `logreg.pkl`
2. `adaboost.pkl`
3. `decision_tree.pkl`
4. `lemmatized_logreg.pkl`
5. `lemmatized_adaboost.pkl`
6. `lemmatized_decision_tree.pkl`

# Data Cleaning
During the project's first iteration and early stages, scores were very high (around 0.98 R2) with minimal data cleaning. This revealed that there was multicollinearity or interdependence within the variables. We used for analysis. The following sections will illustrate the step-by-step process that led to the high-level data-cleaning decisions that we took to address these issues. One step we took was removing medication names from the r/OCD thread and any mention of the terms `OCD,` and `autism` (and their derivative terms) from posts used to train the classification models. This accounted for some of the overfitting in Our analysis, but further analysis would benefit from a more rigorous application of regex and data cleaning to identify new word patterns that could skew results one way or the other. Accuracy can be a misleading metric depending on the project's goals; we may want to optimize for something like an F1 score in the future instead of pure accuracy and comparison to baseline.

## Balancing the Data
To balance the classes between the OCD and autism subreddits, We sequentially dropped values from the majority class until equilibrium was reached. The resulting dataframe contains balanced distributions of posts from the `r/OCD` and `r/Autism` subreddits, respectively.

## User Top Word Analysis
After removing stopwords, We used a combination of `nltk` and python's `re` library to generate a list of keywords for each user. We then used these keywords to create a dictionary of users and their most common keywords.
I used this dictionary to filter the data and remove users that did not have at least ten posts.

```python
# add 'I' to the list of stopwords
# dict of users and their most common word
users_aut = df_aut['author'].value_counts().index
users_aut_most_common_word = {}
for user in users_aut:
user_df = df_aut[df_aut['author'] == user]
user_df = user_df['selftext'].str.split(expand=True).stack().value_counts()[:50].sort_values(ascending=True)
# most common words are in user_df.index, go from most to least common, and pick the first one that is not in the stop words list
for word in user_df.index:
if word not in stopwords_list: # if the word is not in the stop words list
users_aut_most_common_word[user] = word
#print(f'{user} most common word: {word}')
break # break out of the for loop
```
## Keyword Data Leakage
While working through these data-cleaning stages, We found that many users use the names of medications in their posts. This represents a form of data leakage, as certain medications are often indicative of a diagnosis. Due to misspellings and abbreviations, we elected to remove the primary medications used to treat individuals diagnosed with obsessive-compulsive disorder or syndrome. This removed many medication names and other words that were not helpful to the model. To eliminate disparities between the title length and selftext fields, We also combined the two fields into one field. This was done by adding a space between the two fields and concatenating them as the new `selftext`.

## Steps for Data Cleaning in this Study
1. Remove stopwords
2. Remove Medications
3. Combine title and selftext fields into one
4. Remove users with fewer than ten posts
5. Remove punctuation from the text to prepare it for the count vectorizer.


What words are the most frequent unique words in each of the threads?
Create a visual to analyze unique words count versus post length in the `r/Autism` thread to examine vocabulary density and diversity.
Do the same for the `r/OCD` thread.

# Feature Engineering
During feature engineering, we created a new data frame, `df_clean` that contains the following columns:
1. `word_count` - the number of words in each post
2. `unique_word_count` - the number of unique words in each post
3. `post_length` - the length of the post in characters
4. `title_length` - the length of the title in characters
5. `title_word_count` - the number of words in the title
6. `title_unique_word_count` - the number of unique words in the title

# Visualizing the Data

I used a combination of `matplotlib` and `seaborn` to visualize the study's results. We wanted to see how the features were distributed and related to each other. What We was looking for was the presence of outliers and any other anomalies that would need to be addressed before modeling.

# Data Exploration
Title Length Distribution (entire dataset)

![title_length_distribution](./images/title_length_distribution.png)



## Top 25 Users by Post Count in the `r/OCD` Thread

![](./images/top_25_users_ocd_by_posts_with_word.png)

This reveals that there are a large number of posts that have the author deleted, which could mean a deactivated account. This is not a problem for the model as it will not use the author's name as a feature. It would be helpful to remove the posts with the author deleted, though, as they are not valid for the model.

## Top 25 Users by Post Count in the `r/Autism` Thread


![](./images/top_25_users_aut_by_posts_with_word.png)

There is no way to know which deleted accounts are posted and they are all added together. This makes the group very noisy and not valid for the model. Therefore, We will remove these posts from the dataset.

The visual below shows the post frequency by the hour for the two threads. They both pick up in the evening, but OCD has a notable increase in posts early in the day at around 7:00 AM compared to autism which has its minimum at 10:00 AM.

![](./images/post_frequency_by_hour_of_day.png)


I recommend using named entity recognition to identify medication names in the raw text. We specifically would use SpaCy for this. SpaCy was trained on Reddit posts for this purpose and Is a robust analysis tool. We recommend upgrading this to a higher quality large data set once the analysis becomes broader and includes different text sources. If the project's subsequent phases that are beyond this study's scope involved more text, it would be helpful.
We focused on the list of medications below and removed them from the text. This limited data leakage and will also help the model generalize better.
Medications used for OCD are myriad; however, a starting list from medical sources indicates the following.
Clonidine, Quetiapine, Risperidone, Vyvanse, Adderall, Dexedrine, Wellbutrin, Focalin XR, Modafinil, Fluvoxamine, Serzone, Fluvoxamine, Prozac, Lexapro, Paxil, Celexa, Effexor, Zoloft, Cymbalta, Luvox, Pristiq, Remeron, Venlafaxine, Sarafem, Anafranil, Nortriptyline, Tofranil, Xanax, Klonopin, Ativan, Valium, Buspirone, Oxazepam, Aripiprazole, dextroamphetamine, and medications in the SSRI or SNRI families. These include:
Antidepressants – Selective serotonin reuptake inhibitors (SSRIs), such as fluoxetine and paroxetine.
Benzodiazepines – Diazepam, clonazepam, lorazepam, temazepam, alprazolam, chlordiazepoxide, flurazepam, oxazepam, triazolam, divalproex sodium, dronabinol, nabilone, and duloxetine.
** source for data: (Negrini, 2021) **

## Bigrams and Trigrams are great ways to examine text data as well.
Bigrams are two words that are frequently used together. Trigrams are three words that are frequently used together. We can use these to examine the text and see if there are any patterns that we can use to improve the model.

![](./images/top_25_bigrams_ocd.png)


![](./images/top_25_trigrams_ocd.png)

![](./images/top_25_bigrams_aut.png)

![](./images/top_25_trigrams_aut.png)

### Model 1.1. Logistic Regression

```output
Training score: 0.991991643454039
Testing score: 0.9139972144846796
Best score: 0.8992571959145775
Best params: {'cvec__max_df': 0.9, 'cvec__max_features': 3000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1), 'logreg__C': 1, 'logreg__penalty': 'l2'}
Best estimator: Pipeline(steps=[('cvec', CountVectorizer(max_df=0.9, max_features=3000, min_df=2)), ('logreg', LogisticRegression(C=1, solver='liblinear'))])
```

The training score was 99.1%, while the score on the testing data was 91.3%. This meant that the model was good. But. It may be a little bit too good. The first model uses a count vectorizer in conjunction with simple logistic regression, and We utilized a grid search to optimize the model over a set of parameters. The parameters are shown below.

```output
pipe_params_sent_len = {
'cvec__max_features': [1000, 2000, 3000],
'cvec__min_df': [2, 3],
'cvec__max_df': [.9, .95],
'cvec__ngram_range': [(1,1), (1,2)],
'logreg__penalty': ['l1','l2'],
'logreg__C': [1, 2, 3]
}
```

Something to pay attention to in this set of parameters is the penalty terms L1 and L2. These correspond to lasso and ridge regressions.
I also tested N-grams in ranges one to two.
I also tried the same thing with three-fold cross-validation.
I opted to use three-fold cross-validation instead of five-fold optimizing for time's sake. The client wanted results as quickly as possible, and reducing this from five to three seemed the most logical way to meet their expectations.
I ran the model on the full dataset using a grid search, and the results of this model are shown below. A confusion matrix for the logistic regression model revealed that the true positives and negatives are pretty even at 1312 and 1313, respectively. Interestingly, the false positives and negatives are also evenly matched at 123 and 124, respectively.


### Model 1.2. Adaboost

The second model we tested was an AdaBoost model that used logistic regression as its base estimator. Our parameters for the ADA boost model are shown below. We tested 50, 100, and 150 estimators with varying learning rates from .1 up to 1. We also instantiated a grid search on this Adaboost model with three-fold cross-validation, and the results of this model were interesting. The score on the training data was .95 or 95.3% accurate, while the score on the testing data was 91.8% accurate. This is an improvement of 41.8% over the baseline accuracy, but We am still interested to see what further testing can show.

The best parameters were a learning rate of 1 with 150 estimators, 3000 Max features for the count vectorizer, and a `max_df` of 0.90. This model performed the best in the end, outperforming baseline accuracy by 41.8% and outperforming the logistic regression model by 0.50%.

### Model 1.3 Decision Tree
Our third model was the decision tree model. Our best parameters for this model were a max_df of 0.9, 2000 max features for the count vectorizer, and a one-by-two Ngram range. When We first ran this code before, We did a lot of data cleaning and realized there was data leakage. At first, We were getting a training score here of 1.0 and a testing score of 0.9.
The scores on Our training set for the final decision tree model are 99.4% and 81.9% on the testing set. This means that the decision tree model performed 31 percent better than guessing at random.

## Models using lemmatization
Results did not improve enough to warrant using lemmatization. Therefore, we will leave the code in the repository for reference but will not use it in the final model.
The results for the lemmatized models are as follows:

* The Adaboost model scored 89.1% on the testing set and 98.9% on the training set.
* The logistic regression scored 80% on the testing and 99.4% on the training set.
* The decision tree scored 88.9% on the testing and 90.2% on the training set.

Ultimately the logistic regression model was best at classifying posts for these threads, as it could offer insight into inference, while the others tend to be more opaque. Overall, lemmatization did not seem to improve the scores on the testing set though it might have improved the quality of the analysis.

# Comments on Misclassification

Suppose we are applying this model to a clinical application. In that case, we want to make sure that we would optimize for something like recall because recall is focused on reducing the false negatives.

# Conclusions and Recommendations
Our initial hypothesis that a combination of certain NLP techniques, such as sentiment analysis and count vectorization, could be used to build a model that accurately can predict whether a post is from the Autism or OCD subreddit was not sufficiently proved by the study.

Our Alpha Model was good at predicting which subreddit a post belonged to, with accuracy scores between 90 to 99% on the training data. It also scored extremely high on the test data, indicating that some features within the data were overpowering the model and causing it to overfit. One could stop there and be done with their analysis, saying that the model does technically predict which subreddit a post belongs to, but we wanted to see if we could improve the model for the client. For this reason, we moved to the next stage and created our Alpha Models.

Our Beta Models could predict with 91.8% accuracy (on the test set) whether a post was from the Autism or OCD subreddit. This is a significant improvement over the baseline accuracy of 50%. However, it is still not good enough to be used in a production environment. The model is still overfitting, and We believe this is because the data is not clean enough. There are still many words that are irrelevant to the model, and We believe that if We were to clean the data further, We would be able to improve the model.

Our final recommendations are that we number one gather more data and consistently measure these two subreddits to gain a more holistic understanding of what these populations enjoy, what they participate in, what kinds of verbs they use, or nouns they prefer. We would also like to explore one of the features we created that has to do with questions. How many users post questions versus discussions, and are these skewed towards one or the other forum?

There are 8949 unique users in the data frame, and this is a large number when you consider the neuro diversity that exists not only on a spectrum but also on the side of OCD. This study opens many doors for future work and shows promising results though less in the area of linguistics and more in the area of simple prediction.

# Future Work

Future work could be done examining some of the client's requested features, such as word count, unique word count, post link, title, length, title, and word count. For example, title unique word count as well as being able to determine distinctively whether the poster is a parent. Using analysis of words like hyperactive when discussing their children versus themselves.

# Works Cited

https://en.wikipedia.org/wiki/Confusion_matrix

Gedanke. (2018, December 8). Sentiment Analysis of various subreddits [OC] [Reddit Post]. R/Dataisbeautiful. www.reddit.com/r/dataisbeautiful/comments/a4ac4m/sentiment_analysis_of_various_subreddits_oc/

Negrini, G. B. (2021, April 15). Biomedical text natural language processing (BioNLP) using scispaCy. GB Negrini. https://gbnegrini.com/post/biomedical-text-nlp-scispacy-named-entity-recognition-medical-records/

https://stackoverflow.com/a/39308809/12801757

https://stackoverflow.com/a/39308809/12801757

https://stackoverflow.com/questions/24121018/sklearn-gridsearch-how-to-print-out-progress-during-the-execution
