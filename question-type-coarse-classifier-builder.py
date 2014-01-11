#!/usr/bin/env python
"""
Best score: 0.862
Best parameters set:
    clf__alpha: 0.0001
    clf__n_iter: 50
    clf__penalty: 'l2'
    tfidf__norm: 'l2'
    tfidf__use_idf: True
    vect__max_df: 0.75
    vect__ngram_range: (1, 2)
    vect__stop_words: None

"""
__author__ = 'gavin hackeling'
__email__ = 'gavinhackeling@gmail.com'
import os
from time import time
import pickle
from pprint import pprint
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


def grid_search():
    os.chdir('/home/gavin/PycharmProjects/question-type-classifier/corpora/')
    stop_words = [l.strip() for l in open('stop-words.txt', 'rb')]
    categories = ['HUM', 'LOC', 'NUM', 'ENTY', 'DESC', 'ABBR', 'BOOL']
    train = load_files('coarse/',  categories=categories,  shuffle=True,  random_state=42)
    X, y = train.data, train.target

    pipeline = Pipeline([
        ('vect', CountVectorizer(max_df=0.75, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf', SGDClassifier(n_iter=50, penalty='l2')),
    ])

    parameters = {
        #'vect__stop_words': ('english', stop_words, None),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', '2'),
        'clf__alpha': (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001),
        #'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    t0 = time()
    print 'Performing grid search...'
    print 'pipeline:', [name for name, _ in pipeline.steps]
    print 'parameters:'
    pprint(parameters)

    grid_search.fit(X, y)
    print 'done in %0.3fs' % (time() - t0)

    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])


def build_model():
    os.chdir('/home/gavin/PycharmProjects/question-type-classifier/corpora/')
    categories = ['HUM', 'LOC', 'NUM', 'ENTY', 'DESC', 'ABBR', 'BOOL']
    train = load_files('coarse/',  categories=categories,  shuffle=True,  random_state=42)
    X, y = train.data, train.target

    pipeline = Pipeline([
        ('vect', CountVectorizer(max_df=0.75, ngram_range=(1, 2), stop_words=None)),
        ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),
        ('clf', SGDClassifier(n_iter=50, penalty='l2', alpha=0.0001)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, test_size=0.25, random_state=42)
    pipeline.fit(X_train, y_train)
    print 'classifier score:', pipeline.score(X_test, y_test)
    pipeline.fit(X, y)

    filehandler = open('coarse-classifier.p', 'wb')
    pickle.dump(pipeline, filehandler)
    filehandler.close()


if __name__ == '__main__':
    #grid_search()
    build_model()





new = ['Where is the Amazon river located?',
       'Who was the first president',
       'In what state was Columbus born?',
       'What is the fastest animal?',
       'Do you like pie?']