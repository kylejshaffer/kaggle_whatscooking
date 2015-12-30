# Kaggle Competition: What's Cooking

This repo contains my code for entry in my first Kaggle competition - <a href="https://www.kaggle.com/c/whats-cooking">What's Cooking</a>. The code generated a best submission of 133/1388 entries, which is within the top 10% of entries (though this needs to be verified as final results are still being validated).

The code was heavily inspired by an entry for a past Kaggle competition entry by @emanuele: https://github.com/emanuele/kaggle_pbr/blob/master/blend.py. The code builds a stacking model consisting of a: 
* Logistic Regression classifier 
* Bernoulli Naive Bayes
* Linear Support Vector Machine
* Random Forest Classifier
* Extra Trees Ensemble Classifier
* and Bagging Classifier using averaged Logistic Regression classifiers. 

A final Logistic Regression classifier was trained on the class probabilities predicted from each of these component models, and this model was used to make predictions for the final submission. 
