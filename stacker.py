"""
Code for Stacking Idea
Author: Kyle Shaffer
Competition: Kaggle - What's Cooking?

This code is heavily influenced by example code from a past Kaggle entry from @emanuele.
Thanks for posting the code!

Original link here: https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
"""

# Import relevant libraries
from __future__ import division
from datetime import datetime
from pprint import pprint
from tqdm import tqdm
import json, os, numpy as np, scipy as sp, pandas as pd
import cPickle
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.grid_search import GridSearchCV

# Set initial classifiers.
# Each of these classifiers will output class probabilities
# for each of the cuisine classes. Given that there are 6 classifiers,
# the final dataset will consist of 120 (6 * 20 classes) features.
# The logistic regression classifier below will be trained on these
# class probabilities.
# Hyper-parameters for each of the component classifiers were arrived
# at through experimentation on the TFIDF and recipe-length features.
clfs = [LogisticRegression(C=10, class_weight='balanced'),
        BernoulliNB(alpha=0.1),
        BaggingClassifier(base_estimator=LogisticRegression(C=10), n_estimators=100),
        SVC(C=1, kernel='linear', class_weight='balanced', probability=True),
        RandomForestClassifier(n_estimators=200, max_features='auto', class_weight='balanced'),
        ExtraTreesClassifier(n_estimators=200)]

# Helper function to load data quickly.
def load_data():
    # Read in data
    with open('train.json', 'r') as infile:
        train = json.load(infile)
    with open('test.json', 'r') as infile:
        test = json.load(infile)
    # Extract labels
    labels = [i['cuisine'] for i in train]
    labels_dict = {name: idx for idx, name in enumerate(set(labels))}
    y = np.array([labels_dict[label] for label in labels])
    # Extract ingredients
    train_ingredients = [i['ingredients'] for i in train]
    train_ingred_strings = [' '.join(i) for i in train_ingredients]
    test_ingredients = [i['ingredients'] for i in test]
    test_ingred_strings = [' '.join(i) for i in test_ingredients]
    # Extract recipe lengths
    train_lengths = np.array([len(i['ingredients']) for i in train])
    train_lengths = train_lengths.reshape(len(train_ingredients), 1)
    test_lengths = np.array([len(i['ingredients']) for i in test])
    test_lengths = test_lengths.reshape(len(test_ingredients), 1)
    return train_ingred_strings, train_lengths, y, test_ingred_strings, test_lengths, labels_dict

# Function to quickly extract features.
# These features were largely arrived at through experimentation.
def extract_features(train_strings, train_lengths, labels, test_strings, test_lengths, feature_select=True):
    # TFIDF training features
    tfidfvec = TfidfVectorizer(ngram_range=(1,2), stop_words='english', analyzer='word')
    train_tfidf = tfidfvec.fit_transform(train_strings)
    if feature_select:
        feature_selector = SelectKBest(chi2, k=30000)
        train_tfidf = feature_selector.fit_transform(train_tfidf, labels)
    # Concatenate training TFIDF and lengths
    train_features = sp.sparse.hstack((train_tfidf, train_lengths)).tocsr()
    # TFIDF test features
    test_tfidf = tfidfvec.transform(test_strings)
    if feature_select:
        test_tfidf = feature_selector.transform(test_tfidf)
    # Concatenate test TFIDF and lengths
    test_features = sp.sparse.hstack((test_tfidf, test_lengths)).tocsr()
    return train_features, labels, test_features

# Function for generating the train and testing dataset for the final Logistic Regression
# classifier near the end.
def run_stacker(clfs):
    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    shuffle = False
    
    a, b, c, d, e, label_map = load_data()
    X, y, X_submission = extract_features(a, b, c, d, e)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))
   
    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs) * np.unique(y).size))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs) * np.unique(y).size))
    
    row_cnt = 0
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], 20))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            #y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)
            dataset_blend_train[test, row_cnt: row_cnt+20] = y_submission
            dataset_blend_test_j += clf.predict_proba(X_submission)
        dataset_blend_test[:, row_cnt: row_cnt+20] = dataset_blend_test_j / 10.
        row_cnt += 20
    print
    print "Training Logistic Regression classifier."
    # C parameter here set through experimentation.
    clf = LogisticRegression(C=10)
    clf.fit(dataset_blend_train, y)
    # Outputting an initial estimate of performance for the stacking model.
    print "CV 5 Score:", cross_val_score(clf, X=dataset_blend_train, y=y, cv=5).sum()/5.
    # Output the training set, test set, and labels for final submission.
    return dataset_blend_train, y, dataset_blend_test, X, X_submission

# Function for running final predictions. 
# This is mostly to tune the final hyper-parameters
# for the Logistic Regression stacker.
def run_final_predictions(X_train, y_train, X_submit, clf):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 7, 10], 'class_weight': [None, 'balanced']}
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
    grid.fit(X_train, y_train)
    print grid.best_estimator_

    predictions = grid.best_estimator_.predict(X_submit)
    
    with open('test.json', 'r') as infile:
        test = json.load(infile)
    ids = [i['id'] for i in test]
    
    # Format classes for final output
    label_map = load_data()[-1]
    vals = label_map.values()
    rev_labels_dict = dict(zip(vals, label_map.keys()))
    predict_labels = [rev_labels_dict[lab] for lab in predictions]
    
    output = pd.DataFrame(zip(ids, predict_labels))
    output.columns = ['id', 'cuisine']
    print output.cuisine.value_counts()
    output.to_csv('{0}.csv'.format(datetime.now()), index=False)


# Run all the code to generate final output.
# Save features generated from base classifiers along the way.
# This was mostly to facilitate additional experimentation.
if __name__ == "__main__":
    X_train_blend, y_train, X_submit_blend, X_train_base, X_submit_base = run_stacker(clfs)
    with open('X_train_base.pickle', 'w') as outfile:
        cPickle.dump(X_train_base, outfile)
    with open('X_train_blend.pickle', 'w') as outfile:
        cPickle.dump(X_train_blend, outfile)
    with open('X_submit_blend.pickle', 'w') as outfile:
        cPickle.dump(X_submit_blend, outfile)
    with open('X_submit_base.pickle', 'w') as outfile:
        cPickle.dump(X_submit_base, outfile)
    run_final_predictions(X_train_blend, y_train, X_submit_blend, LogisticRegression())


