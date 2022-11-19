
import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import re
import numpy as np
from matplotlib import pyplot as plt



url = 'https://www.reddit.com/r/Autism/'
headers = dict()
headers.update(dict(Accept='application/json', Authorization='Bearer <token>'))
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")
posts = soup.findAll('div', class_="listing-item-container")



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



stop_words = set(stopwords.words("english"))

# Remove Punctuation
def remove_punctuation(text):
    """Remove punctuation from a string"""
    return ''.join(ch for ch in text if ch not in stop_words)

# Lower Case
def lowercase(text):
    """Lower case a string"""
    return text.lower()

df_ocd = pd.read_csv('./data/ocd_thread.csv')
df_autism = pd.read_csv('./data/autism_thread.csv')


lr = LogisticRegression(C=1e6)
lr.fit(df_ocd.drop(columns=('self_text', 'author'), axis=1), df_ocd.target)
accuracy = accuracy_score(lr.predict(df_ocd.drop(columns=('self_text', 'author'), axis=1)), df_ocd.target)
print(f'Accuracy: {accuracy}')


adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
adaboost.fit(df_ocd.drop(columns=('self_text', 'author'), axis=1), df_ocd.target)
accuracy = accuracy_score(adaboost.predict(df_ocd.drop(columns=('self_text', 'author'), axis=1)), df_ocd.target)
print(f'Accuracy: {accuracy}')




dt = DecisionTreeClassifier(random_state=0)
dt.fit(df_ocd.drop(columns=('self_text', 'author'), axis=1), df_ocd.target)
accuracy = accuracy_score(dt.predict(df_ocd.drop(columns=('self_text', 'author'), axis=1)), df_ocd.target)
print(f'Accuracy: {accuracy}')


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



# Plotting the Dendogram
linkage_obj = linkage(distance_func=euclidean)
dendro = linkage_obj.apply(X)
plt.figure()
plt.show()
