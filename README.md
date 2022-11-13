# OCD vs Autism (A Reddit Thread NLP Analysis)
A project by Graham Waters, 2022

---

# Executive Summary

 The end goal for our client is likely a more clinical application of classification to assist users that are seeking help on public forums by using psychoanalysis from text data; however, this is beyond the scope of our initial study. I hope that by learning how these subreddits present linguistically, we may gain insight into the most predictive features that can serve as the first stepping stone toward such a clinical application in the future.

 **Note**: This study is focused solely on linguistic features present in Reddit posts and is not a formal means of diagnosis for identifying autism-spectrum or obsessive-compulsive disorder.

---

# A Table of Contents
- [OCD vs Autism (A Reddit Thread NLP Analysis)](#ocd-vs-autism-a-reddit-thread-nlp-analysis)
- [Executive Summary](#executive-summary)
- [A Table of Contents](#a-table-of-contents)
- [Methods](#methods)
- [About the API](#about-the-api)
- [Data Collection](#data-collection)
- [Files Provided and their Sequence](#files-provided-and-their-sequence)
- [Model Files](#model-files)
- [Data Cleaning](#data-cleaning)
- [Example Austim Post](#example-austim-post)
- [Example OCD Post](#example-ocd-post)
  - [Example 1](#example-1)
  - [Example 2](#example-2)
  - [Balancing the Data](#balancing-the-data)
  - [User Top Word Analysis](#user-top-word-analysis)
  - [OCD and Autism Keywords](#ocd-and-autism-keywords)
  - [Medication Leakage](#medication-leakage)
  - [Steps for Data Cleaning in this Study](#steps-for-data-cleaning-in-this-study)
- [Feature Engineering](#feature-engineering)
    - [Visualizing the Data](#visualizing-the-data)
- [Data Exploration](#data-exploration)
  - [Model 1. Alpha Model](#model-1-alpha-model)
  - [Model 2. Beta Models](#model-2-beta-models)
    - [Model 2.1. Logistic Regression](#model-21-logistic-regression)
    - [Model 2.2. Adaboost](#model-22-adaboost)
    - [Model 2.3 Decision Tree](#model-23-decision-tree)
  - [Models using Lemmatization](#models-using-lemmatization)
- [What posts were misclassified?](#what-posts-were-misclassified)
  - [Misclassified posts](#misclassified-posts)
- [Conclusions and Recommendations](#conclusions-and-recommendations)
- [Future Work](#future-work)
- [Works Cited](#works-cited)
- [Appendix A. - Hyperparameters used for Model 1. Alpha Model.](#appendix-a---hyperparameters-used-for-model-1-alpha-model)
- [Appendix B. - Hyperparameters used for Beta Models.](#appendix-b---hyperparameters-used-for-beta-models)
- [Appendix C. Exemplary posts](#appendix-c-exemplary-posts)
![](./images/3.jpg)

---

# Methods

**Problem Statement:**
A wealthy donor with a track record of philanthropic contributions to both Autism and OCD research organizations contacted our organization, asking for a model that they can utilize to identify post characteristics on Reddit.
The purposes of this study (towards those ends) are to:

1) Use Pushshift API to scrape Reddit posts from the Autism and OCD subreddits.
2) To build a predictive model that can accurately predict whether a post is from the Autism or OCD subreddit

To accomplish these goals, we hypothesize that count vectorization, and Logistic Regression, Adaboost, or Decision Trees can be used to build a model that accurately can predict whether a post is from the Autism or OCD subreddit. Success in this study would mean that our model has a misclassification rate of less than 10% and an accuracy score of greater than 90% on the test data set.

[presentation](https://www.canva.com/design/DAFOTzanP1s/gXPNiG_2EAQ7svxm0Y6dPQ/edit?utm_content=DAFOTzanP1s&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

# About the API

Pushshift's API is relatively straightforward. For example, if I want the posts from [`/r/boardgames`](https://www.reddit.com/r/boardgames), all I have to do is use the following URL: https://api.pushshift.io/reddit/search/submission?subreddit=boardgames

# Data Collection
To gather data for this analysis, I scraped Reddit for posts on two threads, the `r/Autism` thread, and the `r/OCD` thread.

Before getting too deep into the project and eliminating rows/columns from the data, I want to get a bird's eye view of what I have scraped from these subreddits.

We want the data-cleaning process to be as scalable as possible for future research, so I will use a function to clean the data. This function will clean the data for both the `Autism` and `OCD` subreddits.

# Files Provided and their Sequence
The files are ordered as follows:

1. feature_engineering.ipynb # This file contains the code for the feature engineering process.
2. data_cleaning.py # data cleaning functions that I consolidated from a data cleaning notebook for space optimization.
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

This process took the most time and thought out of any of the sections of this study. During the first iteration and early stages of the project, scores were very high (around 0.98 R2) with little to no data cleaning. However, the model was overfitting and was not generalizable to new data. I had to go back and clean the data to get a more generalizable model. I also identified data leakage due to the words `OCD` and `autism` being present in the titles and selftext fields of the posts. I removed these words from the titles etc., and re-ran the model. This resulted in a decrease in the R2 score.

STIC is an acronym that repeatedly appears in autism posts. For future research I would eliminate this from the analysis or add it to the stop words. In addition I would also remove sensory processing disorder and anything to do with sensory systems. It's also possible that terms like stimming and sm, or tantrum Could be overpowered when it comes to a pure word analysis and may lead to overfitting. I accounted for some of this in my analysis but further analysis would benefit from a more rigorous application of regex and data cleaning to identify new word patterns that could be skewing results one way or the other. Accuracy can be a misleading metric and depending on the goals of the project we may want to optimize for something like an F1 score in the future as opposed to pure accuracy and a comparison to baseline.

# Example Austim Post

The following post is an example of a post from the `r/Autism` subreddit. The post is not perfectly cleaned to give an idea of some of the cleaning challenges that exist in this dataset. Any mentions of "autism" explicitly are removed from the examples in the following sections.


```output
"Workplace undergoing renovations and I'm drowning. My workplace workspace is undergoing renovations. They built up floor to ceiling temporary walls around my working area while they renovate the space around us. The walls were supposed to absorb sound while construction is going on but it does nothing. I hear banging, drilling, vacuuming and workers screaming at each other. There are 15 people any any time in probably a 100 year old femaleemalet by 100 year old femaleemalet enclosed space. We had open concept before. I am drowning here and having a breakdown. Everyone is poo-pooing my issues. They tell me it is temporary (it will go on 4-6 months). They tell me I'm not the only have to put up with that. They do not know that I am stic. The walls are keeping the sound in and the air out. I have a lot of sensory dysfunction. It is too hot in here because there is no ventilation (this might be temporary until they work out the airlow) and I'm extremely sensitive to heat, and it is making me extremely dizzy. I am dealing with a lot of extra noise and people. My is through the roof. People eat at their desks and in this enclosed space it is creating a lot of extra scents that, while not unpleasant, is strong and I cannot deal with. The construction is creating extra dust and allergens. I feel that these walls are closing in on me. People pick up the phone constantly and it is amplified by the enclosed walls. The list goes on. I want to take short term disability (my company provides it, full base salary for 3 months) while this construction is going on, but I don't know how I can even get a medical professional to approve it. Now I am a person who has not called in sick a single day in the 20+ years My family doctor refuses to deal with mental health issues and referred me to a shrink which I can only see next year. I currently see a registered social worker for but since she is not a medical professional, she cannot write me a disability letter. I don't know how to get help. I'm in Ontario, Canada if anyone can offer me any advice."
```

# Example OCD Post
## Example 1
```output
'Asymmetrical physical exercises are setting me off. I was recently assessed by a physical therapist for an issue with my pelvis where it is rotated somewhat off kilter. Apparently it means my right side of my hip is lower than my left and the left is tilted backwards while the right is more forwards.
```

## Example 2
```output
"I think i have  I often get a sudden urge to touch, tap or stroke different objects or surfaces(no not that) on specific places or just the whole thing. If i dont do it i get a very wierd tense feeling in my fingertips and palms. I also get the need to beat the insides of my hands nd fingers very often to get the same kind of feeling to go away. Tapping my fingers againts a solid surface in a rythmic way usually helps with the stress from these instances.     Please tell me if there is a diagnosis for this behaviour. None of the people i ask about this know anything about it and think it's weird."
```


## Balancing the Data

To balance the classes between the OCD and autism subreddits, I sequentially dropped values from the majority class until equilibrium was reached. The resulting dataframe contains balanced distributions of OCD and autism.

Build the classification baseline model to predict the majority class. This is how we get this for our classification problem.


## User Top Word Analysis

During the data exploration phase, I process the users in the OCD thread to determine their most common words and the top 50 words used for each author. I then use these values as annotations on a Matplotlib visual shown below.

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



## OCD and Autism Keywords

In the process of working through these data-cleaning stages, I found that many users use the names of medications in their posts. This could also be a form of data leakage as they are often indicative of a diagnosis. It could be worth examining to see if users that are posting on the autism thread use medication names more or less than those on the ocd thread. Should this be considered a form of data leakage?

## Medication Leakage

Due to misspellings and abbreviations, I elected to remove the primary medications used to treat individuals diagnosed with obsessive-compulsive disorder or syndrome. This removed a lot of the medication names and other words that were not helpful to the model.

To eliminate disparities between the length of the title and selftext fields, I also combined the two fields into one field. This was done by adding a space between the two fields and concatenating them together as the new `selftext`.

## Steps for Data Cleaning in this Study
1. Combine title and selftext fields into one field.
2. Generate title-specific features and add them to the dataframe.
   1. Is the title a question?
   2. Are there all-caps words in the title? How many?
   3. What is the length of the title?
   4. What is the title's Sentiment Polarity?
      1. Positive
      2. Negative
      3. Neutral
      4. Compound
3. Before changing the `selftext` field at all run feature_engineering.ipynb to generate features that illustrate reading time... etc.
4. Clean the `selftext` field.
   1. Remove URLs
   2. Remove Special Characters
   3. Remove OCD and Autism from the `selftext` field.
   4. Remove all words related to sex or the activity of physical intimacy, as these are overrepresented in the `r/OCD` thread and will result in overfitting.
      1. The one caveat to this step is that any terms that relate to a sexual orientation such as homosexuality or heterosexuality are not removed as they occur evenly in both subreddits.
    5. Eliminate any medication names from the text as well as dosage amounts. This is done to eliminate data leakage.
    6. Remove all words that are not in the English dictionary.
   5. Save the results to `df_clean.csv` for use in the modeling, and data expl
5. Drop rows with a selftext length of less than ten words.
6. Lowercase all words in the `selftext` field.
7. Instantiate a Lemmatizer and Lemmatize all words in the `selftext` field, save this new field as `selftext_lemmatized`.
8. Instantiate a Stemmer and Stem all words in the `selftext` field; save this new field as `selftext_stemmed`.


What words are the most frequent unique words in each of the threads?
Create a visual to analyze unique words count versus post length in the autism thread to examine vocabulary density and diversity.
Do the same for the `r/OCD` thread.

# Feature Engineering
1. `word_count` - the number of words in each post
2. `unique_word_count` - the number of unique words in each post
3. `post_length` - the length of the post in characters
4. `title_length` - the length of the title in characters
5. `title_word_count` - the number of words in the title
6. `title_unique_word_count` - the number of unique words in the title


It would be interesting to determine which of the two subreddits has a higher percentage of LGBTQ+ terms. This is beyond the scope of this study.

I noticed significant numbers of posters using words like "hyperactive" when discussing their children versus themselves.
* "her/his brother/sister"
* "my son/daughter"
* "my child"
* "as parent*" used by parents.
* "year old" is usually used by parents describing their child.

These are simply interesting observations and are not meant to inform the model. Instead, they simply augment the analysis with context.

During the data cleaning process, I create two data frames, one for the r/ocd posts and the other for the r/autism posts. These are named df_ocd and df_aut respectively. Finally, I loop through the words in the self-text lemmatized field to determine all words that start with I and are less than or equal to four characters in length to append to my stop words list.


### Visualizing the Data

I used a combination of `matplotlib` and `seaborn` to visualize the results of the study. I wanted to see how the features were distributed and how they were related to each other. What I was looking for was the presence of outliers and any other anomalies that would need to be addressed before modeling.

# Data Exploration

Getting into the details of these threads was a fascinating experience. I learned a lot about the kinds of posts one can find in these two forums. After doing background research on the topic I found one paper that coined the term `mood profile` (Gedanke, 2018) to describe a subreddit based on sentiment scores. I decided to use this term to describe the sentiment of the posts in the `r/Autism` and `r/OCD` subreddits.

Title Length Distribution (entire dataset)

![title_length_distribution](./images/title_length_distribution.png)



Top 25 Users by Post Count in the `r/OCD` Thread

![](./images/top_25_users_ocd_by_posts_with_word.png)

This reveals that there are a large number of posts that have the author deleted, which could mean a deactivated account. This is not a problem for the model as it will not use the author's name as a feature. It would be helpful to remove the posts with the author deleted, though, as they are not valid for the model.

Top 25 Users by Post Count in the `r/Autism` Thread


![](./images/top_25_users_aut_by_posts_with_word.png)

There is no way to know which deleted accounts posted, and they are all added together. This makes the group very noisy and not valid for the model. Therefore, I will remove these posts from the dataset.

The visual below shows the post frequency by the hour for the two threads. They both pick up in the evening, but OCD has a notable increase in posts early in the day at around 7:00 AM compared to autism which has its minimum at 10:00 AM.

![](./images/post_frequency_by_hour_of_day.png)


I recommend using named entity recognition as well to identify medication names in the raw text. I specifically would use SpaCy for this. SpaCy was trained on Reddit posts for this very purpose and Is a very robust tool for analysis. I also recommend. upgrading this to the higher quality large data set once the analysis becomes broader and includes different text sources. If the next phases of the project that are beyond the scope of this study involved more text it would be useful.

For now, we will run with the list of medications below and remove them from the text. This will eliminate data leakage and will also help the model generalize better.
Medications used for OCD are myriad; however, a starting list from medical sources indicates the following.
Clonidine, Quetiapine, Risperidone, Vyvanse, Adderall, Dexedrine, Wellbutrin, Focalin XR, Modafinil, Fluvoxamine, Serzone, Fluvoxamine, Prozac, Lexapro, Paxil, Celexa, Effexor, Zoloft, Cymbalta, Luvox, Pristiq, Remeron, Venlafaxine, Sarafem, Anafranil, Nortriptyline, Tofranil, Xanax, Klonopin, Ativan, Valium, Buspirone, Oxazepam, Aripiprazole, dextroamphetamine, and medications in the SSRI or SNRI families. These include:
Antidepressants – Selective serotonin reuptake inhibitors (SSRIs), such as fluoxetine and paroxetine.
Benzodiazepines – Diazepam, clonazepam, lorazepam, temazepam, alprazolam, chlordiazepoxide, flurazepam, oxazepam, triazolam, divalproex sodium, dronabinol, nabilone, and duloxetine.
These will be filtered in the code to eliminate leakage.

(Negrini, 2021)

Bigrams and Trigrams are great ways to examine text data as well.
![](./images/top_25_bigrams_ocd.png)
![](./images/top_25_trigrams_ocd.png)

![](./images/top_25_bigrams_aut.png)
![](./images/top_25_trigrams_aut.png)

## Model 1. Alpha Model

Alpha Model is the first model that I built. It is a simple model that utilizes a pipeline with count vectorization and multiple logistic regression to classify posts from either thread. To optimize the model and "tune" it to give the most accurate results, I utilized sklearn's GridsearchCV package to optimize over a set of hyperparameters.

After my first models were trained, I had a very high r2 score. I revisited the data cleaning stage to narrow down the posts a little bit more before continuing with further iterations of my models.

See [modeling.ipynb](notebooks/modeling.ipynb) for this model


Now that the data has been cleaned I can move on to the next iteration of my model.

## Model 2. Beta Models

### Model 2.1. Logistic Regression

Training score: 0.991991643454039 Testing score: 0.9139972144846796 Best score: 0.8992571959145775 Best params: {'cvec__max_df': 0.9, 'cvec__max_features': 3000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1), 'logreg__C': 1, 'logreg__penalty': 'l2'} Best estimator: Pipeline(steps=[('cvec', CountVectorizer(max_df=0.9, max_features=3000, min_df=2)), ('logreg', LogisticRegression(C=1, solver='liblinear'))])
The training score is 99.1% while the score on the testing data is 91.3%. This means that. The model is good. But. It may be a little bit too good. The first model uses account vectorizer in conjunction with a simple logistic regression and I utilized a grid search to optimize the model over a set of parameters. The parameters are shown below.

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

Something to pay attention to in this set of parameters is the penalty terms L1 and L2 these correspond to lasso and ridge regressions.
I also tested in grams in ranges one to two.


I elected to use three-fold cross-validation instead of 5/4 optimizing for a time. The client wanted results as quickly as possible and this seemed like the most logical way.
A confusion matrix for the logistic regression model reveals that the true positives and true negatives are fairly even at 1312 and 1313 respectively interestingly the false positives and false negatives are also evenly matched at 123 and 124 respectively


### Model 2.2. Adaboost

The second model that I tested was an AdaBoost model that has used logistic regression as its base estimator. My parameters for the ADA boost model are shown below, and I tested 50, 100, and 150 estimators with varying learning rates from .1 up to 1 I also instantiate a grid search on this Adaboost model with three-fold cross-validation and the results of this model were interesting. The score on the training data was .95 or 95.3% accurate while the score on the testing data was 91.8% accurate. This is an improvement of 41.8% over the baseline accuracy but I am still interested to see what further testing can show. The best parameters were a learning rate of 1 with 150 estimators 3000 Max features for the account correct riser and a Max DF for the count vectorizer of 0.90. This model performed the best in the end outperforming baseline accuracy by 41.8% and outperforming the logistic regression model by 0.5%.

### Model 2.3 Decision Tree
My third model was the decision tree model. I ended up with my best parameters for this model being a Max DF of 0.9, 2000 Max features for the count vectorizer and a one by two Ngram range When I first ran this code before I did a lot of data cleaning and realized that there was data leakage. At the first, I was getting a training score here of 1.0 and a testing score of 0.9.
The scores on my training set for the final decision tree model are 99.4% and 81.9% on the testing set. This means that the decision tree model performed 31 percent better than guessing at random.

## Models using Lemmatization

Results did not improve enough to warrant using lemmatization. I will leave the code in the repository for reference but will not be using it in the final model.
The results for the lemmatized models are as follows:

* The adaboost model scored 89.1% on the testing set and 98.9% on the training set.
* The logistic regression scored 80% on the testing and 99.4% on the training set.
* The decision tree scored 88.9% on the testing and 90.2% on the training set.

Ultimately if we are looking to make statements about what kinds of words can lend weight to a post belonging to threads A or B (autism or OCD) and thus desire interpretability we would want to look at the logistic regression model as it could offer insight into inference while the others tend to be more opaque. Overall lemmatizeation did not seem to improve the scores on the testing set though in reality it might have improved the quality of the analysis.

# What posts were misclassified?

## Misclassified posts

If we are applying this model to a clinical application we want to make sure that we would optimize for something like recall because recall it is focused on reducing the false negatives.

# Conclusions and Recommendations
My initial hypothesis that a combination of key NLP techniques such as sentiment analysis and count vectorization, could be used to build a model that accurately is able to predict whether a post is from the Autism or OCD subreddit was not sufficiently proved by the analysis.

Our Alpha Model was good at predicting which subreddit a post belonged to, with accuracy scores around 90 to 99% on the training data. It also scored extremely high on the test data, which was an indicator that some features within the data were overpowering the model and causing it to overfit. One could stop there and be done with their analysis, saying that the model does technically predict which subreddit a post belongs to, but I wanted to see if I could improve the model for scalability. For this reason, I moved the next stage and created my Beta Model.

My Beta Models were able to predict with 91.8% accuracy (on the test set) whether a post was from the Autism or OCD subreddit. This is a significant improvement over the baseline accuracy of 50%. However, it is still not good enough to be used in a production environment. The model is still overfitting, and I believe that this is due to the fact that the data is not clean enough. There are still a lot of words that are not relevant to the model, and I believe that if I were to clean the data further, I would be able to improve the model.

My final recommendations are that we number one gather more data and consistently measure these two subreddits to gain a more holistic understanding of what these populations enjoy, what they participate in, what kinds of verbs they use, or nouns they prefer. I would also really like to explore one of the features that I created that has to do with questions. How many users post questions versus discussions and are these skewed towards one or the other forum?

There are 8949 unique users in the data frame and this is a large number when you consider the neuro diversity that exists not only on a spectrum but also on the side of OCD. This study opens many doors for future work and shows promising results though less in the area of linguistics and more in the area of simple prediction.

# Future Work

Future work could be done examining some of the features that the client requested, such as word count, unique word count, post link, title, length, title, and word count. Title unique word count as well as being able to determine distinctively whether the poster is a parent. Using analysis of words like hyperactive when discussing their children versus themselves.

# Works Cited

https://en.wikipedia.org/wiki/Confusion_matrix

Gedanke. (2018, December 8). Sentiment Analysis of various subreddits [OC] [Reddit Post]. R/Dataisbeautiful. www.reddit.com/r/dataisbeautiful/comments/a4ac4m/sentiment_analysis_of_various_subreddits_oc/

Negrini, G. B. (2021, April 15). Biomedical text natural language processing (BioNLP) using scispaCy. GB Negrini. https://gbnegrini.com/post/biomedical-text-nlp-scispacy-named-entity-recognition-medical-records/

https://stackoverflow.com/a/39308809/12801757

https://stackoverflow.com/a/39308809/12801757

https://stackoverflow.com/questions/24121018/sklearn-gridsearch-how-to-print-out-progress-during-the-execution


# Appendix A. - Hyperparameters used for Model 1. Alpha Model.
```python
pipe_params_sent_len = {
     'cvec__max_features': [1000, 2000, 3000],
     'cvec__min_df': [2, 3],
     'cvec__max_df': [.9, .95],
     'cvec__ngram_range': [(1,1), (1,2)],
     'logreg__penalty': ['l1','l2'],
     'logreg__C': [1, 2, 3]
}

```
# Appendix B. - Hyperparameters used for Beta Models.
```output
AdaBoost Model Fitted
Results for:  GridSearchCV(cv=3,
             estimator=Pipeline(steps=[('cvec', CountVectorizer()),
                                       ('logreg',
                                        LogisticRegression(solver='liblinear'))]),
             n_jobs=-1,
             param_grid={'cvec__max_df': [0.9], 'cvec__max_features': [3000],
                         'cvec__min_df': [2], 'cvec__ngram_range': [(1, 1)],
                         'logreg__C': [1, 2, 3], 'logreg__penalty': ['l2']})
    Training score: 0.9890799256505576
    Testing score: 0.891637630662021
    Best score: 0.8842929080089985
    Best params: {'cvec__max_df': 0.9, 'cvec__max_features': 3000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1), 'logreg__C': 1, 'logreg__penalty': 'l2'}
    Best estimator: Pipeline(steps=[('cvec',
                 CountVectorizer(max_df=0.9, max_features=3000, min_df=2)),
                ('logreg', LogisticRegression(C=1, solver='liblinear'))])
Results for:  GridSearchCV(cv=3,
             estimator=Pipeline(steps=[('cvec', CountVectorizer()),
                                       ('dt', DecisionTreeClassifier())]),
             n_jobs=-1,
             param_grid={'cvec__max_df': [0.9, 0.95],
                         'cvec__max_features': [1000, 2000, 3000],
                         'cvec__min_df': [2, 3],
                         'cvec__ngram_range': [(1, 1), (1, 2)],
                         'dt__max_depth': [None, 2, 3, 4],
                         'dt__min_samples_leaf': [1, 2, 3],
                         'dt__min_samples_split': [2, 3, 4]})
    Training score: 0.9944237918215614
    Testing score: 0.8010452961672474
    Best score: 0.7943773988354832
    Best params: {'cvec__max_df': 0.9, 'cvec__max_features': 2000, 'cvec__min_df': 3, 'cvec__ngram_range': (1, 1), 'dt__max_depth': None, 'dt__min_samples_leaf': 1, 'dt__min_samples_split': 4}
    Best estimator: Pipeline(steps=[('cvec',
                 CountVectorizer(max_df=0.9, max_features=2000, min_df=3)),
                ('dt', DecisionTreeClassifier(min_samples_split=4))])
Results for:  GridSearchCV(cv=3,
             estimator=Pipeline(steps=[('cvec', CountVectorizer()),
                                       ('ada', AdaBoostClassifier())]),
             n_jobs=-1,
             param_grid={'ada__learning_rate': [0.1, 0.5, 1],
                         'ada__n_estimators': [50, 100, 150],
                         'cvec__max_df': [0.9, 0.95],
                         'cvec__max_features': [1000, 2000, 3000],
                         'cvec__min_df': [2, 3],
                         'cvec__ngram_range': [(1, 1), (1, 2)]})
    Training score: 0.9027648698884758
    Testing score: 0.8898954703832752
    Best score: 0.8881271989536108
    Best params: {'ada__learning_rate': 0.5, 'ada__n_estimators': 150, 'cvec__max_df': 0.9, 'cvec__max_features': 3000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1)}
    Best estimator: Pipeline(steps=[('cvec',
                 CountVectorizer(max_df=0.9, max_features=3000, min_df=2)),
                ('ada',
                 AdaBoostClassifier(learning_rate=0.5, n_estimators=150))])
Results for:  GridSearchCV(cv=3,
             estimator=Pipeline(steps=[('cvec', CountVectorizer()),
                                       ('logreg',
                                        LogisticRegression(solver='liblinear'))]),
             n_jobs=-1,
             param_grid={'cvec__max_df': [0.9], 'cvec__max_features': [3000],
                         'cvec__min_df': [2], 'cvec__ngram_range': [(1, 1)],
                         'logreg__C': [1, 2, 3], 'logreg__penalty': ['l2']})
    Training score: 0.9890799256505576
    Testing score: 0.891637630662021
    Best score: 0.8842929080089985
    Best params: {'cvec__max_df': 0.9, 'cvec__max_features': 3000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1), 'logreg__C': 1, 'logreg__penalty': 'l2'}
    Best estimator: Pipeline(steps=[('cvec',
                 CountVectorizer(max_df=0.9, max_features=3000, min_df=2)),
                ('logreg', LogisticRegression(C=1, solver='liblinear'))])
Results for:  GridSearchCV(cv=3,
             estimator=Pipeline(steps=[('cvec', CountVectorizer()),
                                       ('dt', DecisionTreeClassifier())]),
             n_jobs=-1,
             param_grid={'cvec__max_df': [0.9, 0.95],
                         'cvec__max_features': [1000, 2000, 3000],
                         'cvec__min_df': [2, 3],
                         'cvec__ngram_range': [(1, 1), (1, 2)],
                         'dt__max_depth': [None, 2, 3, 4],
                         'dt__min_samples_leaf': [1, 2, 3],
                         'dt__min_samples_split': [2, 3, 4]})
    Training score: 0.9939591078066915
    Testing score: 0.802439024390244
    Best score: 0.7946097678374583
    Best params: {'cvec__max_df': 0.9, 'cvec__max_features': 2000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1), 'dt__max_depth': None, 'dt__min_samples_leaf': 1, 'dt__min_samples_split': 4}
    Best estimator: Pipeline(steps=[('cvec',
                 CountVectorizer(max_df=0.9, max_features=2000, min_df=2)),
                ('dt', DecisionTreeClassifier(min_samples_split=4))])
Results for:  GridSearchCV(cv=3,
             estimator=Pipeline(steps=[('cvec', CountVectorizer()),
                                       ('ada', AdaBoostClassifier())]),
             n_jobs=-1,
             param_grid={'ada__learning_rate': [0.1, 0.5, 1],
                         'ada__n_estimators': [50, 100, 150],
                         'cvec__max_df': [0.9, 0.95],
                         'cvec__max_features': [1000, 2000, 3000],
                         'cvec__min_df': [2, 3],
                         'cvec__ngram_range': [(1, 1), (1, 2)]})
    Training score: 0.9027648698884758
    Testing score: 0.8898954703832752
    Best score: 0.8881271989536108
    Best params: {'ada__learning_rate': 0.5, 'ada__n_estimators': 150, 'cvec__max_df': 0.9, 'cvec__max_features': 3000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1)}
    Best estimator: Pipeline(steps=[('cvec',
                 CountVectorizer(max_df=0.9, max_features=3000, min_df=2)),
                ('ada',
                 AdaBoostClassifier(learning_rate=0.5, n_estimators=150))])

```

# Appendix C. Exemplary posts

```output

Does Anyone Else Have This? hello. i have a problem, or a question, and i want to know if this is  or something else, or if anyone else happens to experience this.    i have some sort of obsession with dividing sentences, i guess you could say. i will take a phrase and demonstrate.    “i love you”  now, what my brain does is choose either 3, 4, 5, 7, or 11 and start counting.  if i dont want to include the spaces, i use 4. i-l-o-v; e-y-o-u. if i want the spaces, 5. i- -l-o-v; e- -y-o-u. and it’s evenly divided now. this will happen with almost every sentence, phrase, word, etc. that i come across. please tell me if this has a name. thank you

```
