
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
Results for: GridSearchCV(cv=3,
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
Results for: GridSearchCV(cv=3,
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
Results for: GridSearchCV(cv=3,
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
Results for: GridSearchCV(cv=3,
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
Results for: GridSearchCV(cv=3,
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
Results for: GridSearchCV(cv=3,
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

Does Anyone Else Have This? hello. We have a problem, or a question, and We want to know if this is something else, or if anyone else happens to experience this. We have some sort of obsession with dividing sentences, We guess you could say. We will take a phrase and demonstrate. "i love you" now, what Our brain does is choose either 3, 4, 5, 7, or 11 and start counting. if We dont want to include the spaces, We use 4. i-l-o-v; e-y-o-u. if We want the spaces, 5. i- -l-o-v; e- -y-o-u. and it's evenly divided now. this will happen with almost every sentence, phrase, word, etc. that We come across. please tell me if this has a name. thank you

```
