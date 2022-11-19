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
  - [Data Collection](#data-collection)
    - [Reddit API](#reddit-api)
    - [Keywords](#keywords)
  - [Feature Engineering](#feature-engineering)
    - [Text Preprocessing](#text-preprocessing)
    - [Model Building](#model-building)
      - [Logistic Regression](#logistic-regression)
      - [Adaboost](#adaboost)
      - [Decision Tree](#decision-tree)
      - [Keyword Vectorizer](#keyword-vectorizer)
    - [Visualization](#visualization)
  - [Results](#results)
  - [References](#references)


---
# Methods
## Data Collection
### Reddit API
We collected data from the `r/Autism` and `r/OCD` subreddits using the Reddit API. We used the Python library `requests` to make requests to the API and parse the JSON response. The following code snippet shows the request and parsing steps.
```python
import requests
from bs4 import BeautifulSoup

url = 'https://www.reddit.com/r/Autism/'
headers = dict()
headers.update(dict(Accept='application/json', Authorization='Bearer <token>'))
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")
posts = soup.findAll('div', class_="listing-item-container")
```
The `BeautifulSoup` library was used to parse the HTML content of the page. We then iterated over the posts and extracted the post title, link, and body.
### Keywords
We also collected keywords from the posts. We did this by searching for the keyword in the post's text. If the keyword was found, we appended it to a list.
```python
def get_keywords(post):
    """Get the keywords from a post"""
    # Get the keywords from the post
    keywords = set()
    for word in re.split("\W+", post.text):
        if word in keywords:
            continue
        else:
            keywords.add(word)
    return keywords
```
## Feature Engineering
### Text Preprocessing
We preprocessed the text data by removing punctuation and lower casing the words. We also removed stop words and added them to the stop words list.
```python
stop_words = set(stopwords.words("english"))

# Remove Punctuation
def remove_punctuation(text):
    """Remove punctuation from a string"""
    return ''.join(ch for ch in text if ch not in stop_words)

# Lower Case
def lowercase(text):
    """Lower case a string"""
    return text.lower()
```
### Model Building
#### Logistic Regression
Logistic regression is a binary classification model that uses log odds as its output. In this study, we used the `scikit-learn` library to build a logistic regression model. The following code snippet shows how we built our model.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(C=1e6)
lr.fit(df_ocd.drop(columns=('self_text', 'author'), axis=1), df_ocd.target)
accuracy = accuracy_score(lr.predict(df_ocd.drop(columns=('self_text', 'author'), axis=1)), df_ocd.target)
print(f'Accuracy: {accuracy}')
```
#### Adaboost
Adaboost is an ensemble learning algorithm that combines multiple weak learners into a strong learner. It is often used in conjunction with logistic regression. The following code snippet shows how we built our adaboost model.
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
adaboost.fit(df_ocd.drop(columns=('self_text', 'author'), axis=1), df_ocd.target)
accuracy = accuracy_score(adaboost.predict(df_ocd.drop(columns=('self_text', 'author'), axis=1)), df_ocd.target)
print(f'Accuracy: {accuracy}')
```
#### Decision Tree
Decision trees are a type of tree-based machine learning algorithm that are commonly used in classification problems. They are useful because they are easy to understand and interpret. The following code snippet shows how we built our decision tree model.
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier(random_state=0)
dt.fit(df_ocd.drop(columns=('self_text', 'author'), axis=1), df_ocd.target)
accuracy = accuracy_score(dt.predict(df_ocd.drop(columns=('self_text', 'author'), axis=1)), df_ocd.target)
print(f'Accuracy: {accuracy}')
```
#### Keyword Vectorizer
In order to extract features from the text data, we first need to tokenize the text. This can be done using the `sklearn` library's `CountVectorizer`. We then passed the tokens to the `TfidfTransformer` to generate tf-idsf vectors.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_ocd.drop(columns=('self_text', 'author'), axis=1).astype(str))
tf_vocab = Counter().most_common(len(stop_words)+5)
for i in range(10):
    print(i, len(set(tokens)))
    for j, k in enumerate(tf_vocab):
        if k >= 5:
            break
        X_k = np.array(np.where(X == k))
        X_j = np.zeros((0, 0))
        for x in X_k:
            X_j += 1 * x
        X = np.vstack((X, X_j))
    print(i + 1, len(set(tokens)))
    for j, k in enumerate(tf_vocab):
        if k > 4:
            break
        X_k = np.array(np.where(X == k))
        X_j = np.zeros((0, 0))
        for x in X_k:
            X_j += 1 * x
        X = np.hstack((X, X_j))
        print(i + 2, len(set(tokens)))
```
### Visualization
After building the models, we plotted the results. For the logistic regression model, we created a bar plot showing the predicted probability of being OCD versus the actual value. The following code snippet shows how we generated the figure.
```python
plt.figure()
ax = plt.subplot(111)
x = df_ocd.target.values
y = lr.predict(x)
bar_width = .35
color_map = sns.light_palette("Greens", 10)
colors = color_map.as_hex()
labels = list(range(len(df_ocd.target.index)))
rects = ax.bar(labels, y, width=bar_width, label='Predicted Probability', edgecolor=None, align="center")
ax.legend(loc="best")
ax.set_yticks(np.arange(0, 1.05, .25))
ax.set_xticklabels(list(df_ocd.target.index))
ax.invert_yaxis()
ax.set_title("Logistic Regression Predictions")
fig = plt.gcf()
fig.tight_layout()
```
For the adaboost model, we used a confusion matrix to show the performance of the classifier. The following code snippet shows how we generated the figure.
```python
import numpy as np
from matplotlib import pyplot as plt
confusion_matrix = pd.crosstab(df_ocd.target, df_ocd.prediction)
cm = confusion_matrix(df_ocd.target, df_ocd.prediction)
num_classes = cm.sum(1).max()+1
class_names = list(range(num_classes))
row_positions = np.argsort(-df_ocd.target)
col_indices = np.argpartition(df_ocd.target, -df_ocd.target.size-1)
for row_number, col_name in zip(row_positions, class_names):
    fig = plt.figure()
    ax = plt.subplot(2, num_classes, row_number)
    cnt = conf_matrix.iloc.get_value(row_position=row_number, column_label=col_name)
    ax.barh(class_names, cnt)
    ax.set_ylabel(col_name)
    ax.set_xlim(0, num_classes)
    ax.set_xticks(np.arange(0, num_classes, 1))
    ax.grid()
    plt.savefig('images/logreg_confusion_matrix.png')
    plt.close()
```
Finally, for the decision tree model, we used a dendrogram to visualize the structure of the tree. The following code snippet shows how we generated the figure.
```python
# Plotting the Dendogram
from scipy.cluster.hierarchy import linkage
linkage_obj = linkage(distance_func=euclidean)
dendro = linkage_obj.apply(X)
plt.figure()
plt.show()
```
## Results
The final results are presented below.
<img src="/docs/assets/results.png" alt="Results">


## References
* Gedanke, J., & König, S. (2018). Mood profiles of autism and obsessive compulsive disorder subgroups on reddit. Retrieved from https://www.reddit.com/r/MoodProfiles/comments/8nqw7b/mood_profiles_of_autism_and_obsessive_compulsive/
* Hsu, C.-Y., Lin, Y.-H., Wang, P.-J., Chen, W.-C., Tsai, M.-L., Huang, L.-W., ... & Chiu, E. (2017). A survey on deep learning methods applied to natural language processing tasks. arXiv preprint arXiv:1704.01375.
