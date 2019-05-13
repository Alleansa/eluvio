# Eluvio DS NLP problem

## The problem

The csv file contains 8 columns: created date, created time, up vote, down vote, title, author, over 18 and category.

After simple checking, it can be found that all the news belong to one category (world news) and the down votes are all zero. Thus, it will be interesting to explore the relationship with the up vote and other covariates.
Additionally, 85838 authors are included. 


## Steo 0: Data Processing

## Step 1: the Universal Sentence Encoding

## Step 2: Deep Neural Network Classifier

## Current result

## Some Idea

- So far, the classification results is not good enough. I think this is mainly because the universal sentence encoder can not efficiently extract the information from the titles. A good way to improve this is to use some NLP models to learning the sentence vectors for these titles.

- Due to the limited computation resource, I didn't use the overparameterized dnn model. But I think it will reduce the generalization error by increasing the parameter number.

--
