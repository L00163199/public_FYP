"""
Student Name: Dylan Walsh
Student Number: L00163199
Description: The purpose of this file is to
perform sentiment analysis on a given set of
tweets and to then train a naive bayes model
for better accuracy for future sentiment analysis
on new tweets, given that this project
is ever-changing in nature and continued analysis
would be required

"""

import pandas as pd
import numpy as np

# Vader is particularly useful for sentiment analysis
# with social media, it also doesn't require training
# a machine learning model prior to carrying the analysis out
from scipy.stats import ttest_ind, fisher_exact
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the two datasets
irish_unity_df = pd.read_csv\
    ('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\data_preparation\\irish_unity_tweets_CLEAN.csv')
sinn_fein_df = pd.read_csv\
    ('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\data_preparation\\sinn_fein_tweets_CLEAN.csv')

tweet_sentiment = SentimentIntensityAnalyzer()

# Generates sentiment score for each tweet
def get_sentiment_score(tweet_text_field):
    return tweet_text_field.apply(lambda x: tweet_sentiment.polarity_scores(str(x))['compound'])

# Classifies a given sentiment score into
# positive, negative or neutral based on
# set threshold values
# Note - threshold values for sentiment analysis
# are often subjective and vary depending on
# the data used but the following link(s) 
# were used as a reference:
# https://www.researchgate.net/publication/350761338_A_Review_on_Lexicon-Based_and_Machine_Learning_Political_Sentiment_Analysis_Using_Tweets
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9910766/
def classify_sentiment_score(sentiment_score):
    try:
        sentiment_score = float(sentiment_score)
        if sentiment_score > 0.05:
            return 'positive'
        elif sentiment_score < -0.05:
            return 'negative'
        else:
            return 'neutral'
        
    # Catches exception if sentiment_score
    # is not a valid number
    except ValueError:
        raise ValueError("The sentiment score was not a number between -1 and 1.")
    
# First the sentiment scores will be generated
# for each of the cleaned texts as a result of
# the different pipelines and then the labels, 
# this will be useful later for analysis of how 
# the pipeline(s) has impacted the results
irish_unity_df['sentiment_pipeline1'] = get_sentiment_score(irish_unity_df['cleaned_tweet_text_pipeline1'])
irish_unity_df['sentiment_label_pipeline1'] = \
    irish_unity_df['sentiment_pipeline1'].apply(classify_sentiment_score)

# Generating the scores and labels for
# the cleaned text with pipeline 2
irish_unity_df['sentiment_pipeline2'] = get_sentiment_score(irish_unity_df['cleaned_tweet_text_pipeline2'])
irish_unity_df['sentiment_label_pipeline2'] = \
    irish_unity_df['sentiment_pipeline2'].apply(classify_sentiment_score)

# Sinn Fein dataset
# Pipeline 1
sinn_fein_df['sentiment_pipeline1'] = get_sentiment_score(sinn_fein_df['cleaned_tweet_text_pipeline1'])
sinn_fein_df['sentiment_label_pipeline1'] = \
    sinn_fein_df['sentiment_pipeline1'].apply(classify_sentiment_score)

# Pipeline 2
sinn_fein_df['sentiment_pipeline2'] = get_sentiment_score(sinn_fein_df['cleaned_tweet_text_pipeline2'])
sinn_fein_df['sentiment_label_pipeline2'] = \
    sinn_fein_df['sentiment_pipeline2'].apply(classify_sentiment_score)

# Save the datasets to the current directory
sinn_fein_df.to_csv\
    ('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\model_training_evaluation\\sinn_fein_tweets_SENTIMENT.csv', index=False)
irish_unity_df.to_csv\
    ('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\model_training_evaluation\\irish_unity_tweets_SENTIMENT.csv', index=False)

# Now a naive bayes model can be trained for
# sentiment analysis for future tweets
# Note - multinonomial distribution naive bayes
# cannot contain negative values and therefore
# the negative sentiment scores will need to be
# transformed

# Merge the 2 datasets to produce
# a larger cohort of training data
merged_dataset = pd.concat([irish_unity_df, sinn_fein_df], ignore_index=True)

# Transform the sentiment scores
merged_dataset['sentiment_pipeline2'] = np.exp(merged_dataset['sentiment_pipeline2'])

# Split into training and test
train_data, test_data, train_labels, test_labels =\
      train_test_split(merged_dataset['sentiment_pipeline2'],\
                        merged_dataset['sentiment_label_pipeline2'], test_size=0.3, random_state=42)

# Train the classifier
naive_classifier = MultinomialNB()
naive_classifier.fit(train_data.values.reshape(-1, 1), train_labels)

# Use test set for predictions
naive_predict = naive_classifier.predict(test_data.values.reshape(-1, 1))

# Checking accuracy of the nb classifier
# At time of execution the accuracy of this
# is: 0.4053333333333333, this could be due to
# the training data being small or due to the
# language used in the tweets (hiberno english slang)
print("Naive Bayes Classifier accuracy:", \
      accuracy_score(test_labels, naive_predict))


# Further evaluating model performance
prec_score = precision_score(test_labels, naive_predict, average='weighted')
rec_score = recall_score(test_labels, naive_predict, average='weighted')
f1 = f1_score(test_labels, naive_predict, average='weighted')

print("Precision Score Value: ", prec_score)
print("Recall Score Value: ", rec_score)
print("F1 Score Value: ", f1)


# Outputting the final dataset

# Drop the columns related to pipeline 1
irish_unity_df = irish_unity_df.drop(['sentiment_pipeline1', 'sentiment_label_pipeline1', 'cleaned_tweet_text_pipeline1'], axis=1)
sinn_fein_df = sinn_fein_df.drop(['sentiment_pipeline1', 'sentiment_label_pipeline1', 'cleaned_tweet_text_pipeline1'], axis=1)

# Rename the columns as appropriate
irish_unity_df = irish_unity_df.rename(columns={'cleaned_tweet_text_pipeline2': 'cleaned_tweet_text', 
                                                'sentiment_pipeline2': 'cleaned_tweet_sentiment_score', 
                                                'sentiment_label_pipeline2': 'cleaned_tweet_sentiment_label'})

# Since both datasets have the same column
# names, a suffix can be added to differentiate
irish_unity_df = irish_unity_df.add_suffix('_unity')

sinn_fein_df = sinn_fein_df.rename(columns={'cleaned_tweet_text_pipeline2': 'cleaned_tweet_text', 
                                            'sentiment_pipeline2': 'cleaned_tweet_sentiment_score', 
                                            'sentiment_label_pipeline2': 'cleaned_tweet_sentiment_label'})
sinn_fein_df = sinn_fein_df.add_suffix('_sf')

# Concatenate the dataframes
merged_df = pd.concat([irish_unity_df, sinn_fein_df], axis=1)

# Output to CSV file
merged_df.to_csv\
    ('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\model_training_evaluation\\irishUnity_sinnFein_cleaned_merged.csv', index=False)

# Performing the t-test to address
# the null hypothesis
t_statistic, p_value = ttest_ind(merged_df['cleaned_tweet_sentiment_score_unity'],\
                                  merged_df['cleaned_tweet_sentiment_score_sf'])

# Print the t-statistic and p-value
print(" Performing the t-test")
print("T-statistic: ", t_statistic)
print("P-value: ", p_value)
