#!/usr/bin/env python
"""
done in 2539.012s
Best score: 0.936
Best parameters set:
    clf__alpha: 0.0001
    clf__n_iter: 80
    clf__penalty: 'elasticnet'
    tfidf__norm: 'l2'
    tfidf__use_idf: False
    vect__max_df: 0.75
    vect__max_features: None
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
    categories = ['address', 'artificial', 'bodwat', 'city', 'continent', 'country', 'desert', 'island', 'mountain',
                  'other', 'planet', 'restaurant', 'river', 'sea', 'state', 'vague', 'airport', 'biome',
                  'celestial_body', 'constellation', 'coordinate', 'county', 'direction', 'lake', 'ocean', 'park',
                  'region', 'retail', 'school', 'source', 'street', 'web_address']
    train = load_files('fine/LOC',  categories=categories,  shuffle=True,  random_state=42)
    X, y = train.data, train.target

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    parameters = {
        'vect__stop_words': ('english', stop_words, None),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        'clf__n_iter': (10, 50, 80),
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
    categories = ['address', 'artificial', 'bodwat', 'city', 'continent', 'country', 'desert', 'island', 'mountain',
                  'other', 'planet', 'restaurant', 'river', 'sea', 'state', 'vague', 'airport', 'biome',
                  'celestial_body', 'constellation', 'coordinate', 'county', 'direction', 'lake', 'ocean', 'park',
                  'region', 'retail', 'school', 'source', 'street', 'web_address']
    train = load_files('fine/LOC',  categories=categories,  shuffle=True,  random_state=42)
    X, y = train.data, train.target

    pipeline = Pipeline([
        ('vect', CountVectorizer(max_df=0.75, ngram_range=(1, 2), stop_words=None)),
        ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
        ('clf', SGDClassifier(n_iter=80, penalty='elasticnet', alpha=0.0001)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, test_size=0.25, random_state=42)
    pipeline.fit(X_train, y_train)
    print 'classifier score:', pipeline.score(X_test, y_test)
    pipeline.fit(X, y)

    filehandler = open('fine-loc-classifier.p', 'wb')
    pickle.dump(pipeline, filehandler)
    filehandler.close()


if __name__ == '__main__':
    grid_search()
    #build_model()
