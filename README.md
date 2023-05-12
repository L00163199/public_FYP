# Final Year Project - Dylan Walsh

## Introduction
This is a repository for my final year project. The project focuses on using exploratory and predictive analysis techniques in conjunction with machine learning algorithms to investigate the likelihood of Irish Unification.

## File Directory Structure
Please follow the structure in order, as this is the same order followed during development
- [Data Collection](#data-collection)
- [Data Preparation](#data-preparation)
- [Model Training Evaluation](#model-training-evaluation)
- [Data Analysis](#data-analysis)
- [Web App](#web-app)

## General Note
All folders will have .csv files in them relating to the process they were generated from.
For example, 'data_collection' will have the raw data mined from Twitter without having pre-processing applied to it.
'data_preparation' will have the cleaned and pre-processed csv files.
'model_training_evaluation' will have the csv files appended with sentiment analysis scores/labels

## Data Collection
This folder relates to the data collection aspect of the project. In it's .py files you will find code to mine a sum of tweets over the last 10 years.
User demographics are extracted in this process prior to the pre-processing stage as the regex patterns used as required to match demographic data.
To summarise, once a tweet discussing Irish Unity or Sinn Fein is found the user's profile bio is examined for demographic information. If the required
demographics are found then the tweet and details will be stored.
Note - a 'mine_news_headlines" file is present in this folder. It contains code to mine a sum of news headlines mentioning Irish Unity or Sinn Fein.
However, due to immediate restriction from the API, these were not saved.

## Data Preparation
This folder relates to the data preparation aspect of the project. In it's .py file you will find code to clean and pre-process the data using 2 different pipelines. Both pipelines run the same techniques except for 1. The first pipeline uses stemming for its performance efficiency with large datasets, named 'efficiency_pipeline'. The second pipeline uses lemmatisation, which is slower but more accurate, named 'accuracy_pipeline'. The contents of the mined Tweets will be cleaned and then 2 newly added columns will be appended to the datasets to account for the cleaned text from each pipeline.

## Model Training Evaluation
This folder relates to performing sentiment analysis on the tweets, training the Naive Bayes model for future sentiment classification of tweets and then evalates it's accuracy through train and test, as well as using other evaluation metrics such as the F1 score. The generated sentiment scores and labels will be appended to the datasets.

## Data Analysis
This folder relates to the data analysis part of the project. Now that the dataset has been pre-processed/cleaned and has the sentiment scores/labels added, exploratory and predictive analysis is carried out. Wordclouds, network diagrams, time series analysis, time series forecast, box plots, bar charts, scatter graphs, pearson's coefficient correlation are all performed in this file. Note - this is the final phase of the 'offline' or 'initial' phase of the project. That is, before the web app is made.

## Web App
Note - The web app focuses on functionality moreso than design

This folder relates to the second part of the project - the development of the web application. It is written using flask and follows the general directory layout of Flask projects:
1. Static - where our static files go, such as css and Javascript
2. Templates - where our html files go, including index.html which is our base file of which the others extend from
3. Uploads - where the files the end user uploads will be stored locally, also stores any generated visualisations
4. app.py - holds all core functionality in the web app. The code handles all visualisations and analysis techniques

