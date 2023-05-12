"""
Student Name: Dylan Walsh
Student Number: L00163199
Description: The purpose of this file is to
apply some basic exploratory functions to understand
the dataset(s) prior to being cleaned and preprocessed

"""

import pandas as pd
import nltk
import re
import unicodedata
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

# These datasets are loaded locally
# Can use a file dialogue menu in future for other users
irish_unity_df = \
    pd.read_csv('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\data_collection\\irish_unity_tweets_RAW.csv')
sinn_fein_df = \
    pd.read_csv('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\data_collection\\sinn_fein_tweets_RAW.csv')

# Print the number of unique values in each column in each dataset
print(f"Number of unique values in each column of 'irish_unity_tweets_RAW.csv':\n{irish_unity_df.nunique()}")
print(f"Number of unique values in each column of 'sinn_fein_tweets_RAW.csv':\n{sinn_fein_df.nunique()}")

# Print the top 10 most common words in the tweet text for each dataset

common_words = set(stopwords.words('english'))


# Gets the 20 most commonly used words
# in a given corpus of text
def get_common_words(df, col_name):
    # Creates a set of stop words for filtering
    # out upon search for speed
    stop_words = set(stopwords.words('english'))

    # Uses list comprehension to generate the list
    # of words, then for every tweet it will verify
    # if it is a string and then splits it into
    # individual words and then only keeps words
    # that are not present in the set of stop words
    # or are less than 3 in length as these are likely
    # not words of any meaningful value
    words = [word.lower() for tweet in df[col_name]\
              if isinstance(tweet, str) for word in tweet.split()\
                  if word.lower() not in stop_words and len(word) > 2]
    
    # Words are returned sorted based on value
    # in descending order of the 20 most commonly
    # used words, with the lambda function getting
    # the count value for each word
    return sorted(Counter(words).items(), key=lambda x: x[1], reverse=True)[:20]

print(f"The 20 most commonly used words in tweets mentioning a united Ireland\
       were:\n{get_common_words(irish_unity_df, 'tweet_text')}")
print(f"The 20 most commonly used words in tweets mentioning Sinn Fein\
      were':\n{get_common_words(sinn_fein_df, 'tweet_text')}")

# Cleaning and preprocessing step

# Preprocessing/cleaning pipeline 1
# This pipeline will incorporate both cleaning
# and preprocessing techniques with emphasis on
# efficiency through compression of data using
# stemming and vectorisation
def efficiency_pipeline(tweet_text):
    tweet_text = tweet_text.lower()

    # Removes any URLs from the tweets,
    # this is common with news articles
    # often being referenced in more political topics
    # such as Sinn Fein and Irish Unity
    tweet_text = re.sub(r'http\S+', '', tweet_text)

    # Removes special characters including diacritic marks,
    # i.e. fadas in this case
    tweet_text = unicodedata.normalize('NFKD', tweet_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Removing punctuation, doesn't necessarily
    # provide anything meaningful to text
    tweet_text = re.sub(r'[^\w\s]', '', tweet_text)

    # Removing stop words for better analysis,
    # better to derive more meaningful words
    # than the likes of "the" and "of" etc
    stop_words = set(stopwords.words('english'))

    # Tokenisation
    single_words = tweet_text.split()
    single_words = [word for word in single_words if word not in stop_words]

    # Application of stemming should reduce
    # the number of unique words, may be issues
    # with accuracies of words once in base form
    word_stemmer = PorterStemmer()
    single_words = [word_stemmer.stem(word) for word in single_words]

    # Concatenate all the words together and return them
    full_tweet_text = ' '.join(single_words)

    return full_tweet_text

# Define a function to clean the tweet text
def accuracy_pipeline(tweet_text):
    # Will mostly follow the same steps as
    # the pipeline focusing on efficiency
    
    # Lowercase the text
    tweet_text = tweet_text.lower()
    # Remove URLs
    tweet_text = re.sub(r'http\S+', '', tweet_text)
    # Remove special characters and diacritic marks
    tweet_text = unicodedata.normalize('NFKD', tweet_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove punctuation
    tweet_text = re.sub(r'[^\w\s]', '', tweet_text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    single_words = tweet_text.split()
    single_words = [word for word in single_words if word not in stop_words]
    # Apply lemmatization to the words
    lemmatizer = WordNetLemmatizer()
    single_words = [lemmatizer.lemmatize(word) for word in single_words]
    # Join the words back into a string
    full_text_tweet = ' '.join(single_words)
    return full_text_tweet

# Applying both pipelines to the sinn fein dataset
sinn_fein_df['tweet_text'] = sinn_fein_df['tweet_text'].fillna('')
sinn_fein_df['cleaned_tweet_text_pipeline1'] = \
    sinn_fein_df['tweet_text'].apply(efficiency_pipeline)

sinn_fein_df['cleaned_tweet_text_pipeline2'] = \
    sinn_fein_df['tweet_text'].apply(accuracy_pipeline)

# Applying both pipelines to the Irish unity dataset
irish_unity_df['tweet_text'] = irish_unity_df['tweet_text'].fillna('')
irish_unity_df['cleaned_tweet_text_pipeline1'] = \
    irish_unity_df['tweet_text'].apply(efficiency_pipeline)

irish_unity_df['cleaned_tweet_text_pipeline2'] = \
    irish_unity_df['tweet_text'].apply(accuracy_pipeline)

# Vectorising the cleaned text
# Note - need to create a new TfidVectorizer
# object each time to ensure each dataset
# is appropriately vectorised with its
# own vocabulary and IDF values

# Vectorising each set of cleaned text 
# from the sinn fein dataset
sf_vectoriser_pipe1 = TfidfVectorizer()
sinnFein_pipeline1 = \
    sf_vectoriser_pipe1.fit_transform(sinn_fein_df['cleaned_tweet_text_pipeline1'])

sf_vectoriser_pipe2 = TfidfVectorizer()
sinnFein_pipeline2 = \
    sf_vectoriser_pipe2.fit_transform(sinn_fein_df['cleaned_tweet_text_pipeline2'])

# Vectorising each set of cleaned text
# from the irish unity dataset
unity_vectoriser_pipe1 = TfidfVectorizer()
irishUnity_pipeline1 = \
    unity_vectoriser_pipe1.fit_transform(irish_unity_df['cleaned_tweet_text_pipeline1'])

unity_vectoriser_pipe2 = TfidfVectorizer()
irishUnity_pipeline2 = \
    unity_vectoriser_pipe2.fit_transform(irish_unity_df['cleaned_tweet_text_pipeline2'])

# Evaluating results from both pipelines
# with the sinn fein dataset
print(f"POST PROCESSING PIPELINE 1:The 20 most commonly used words in tweets\
      mentioning Sinn Fein were':\n{get_common_words(sinn_fein_df, 'cleaned_tweet_text_pipeline1')}")

print(f"POST PROCESSING PIPELINE 2:The 20 most commonly used words in tweets\
      mentioning Sinn Fein were':\n{get_common_words(sinn_fein_df, 'cleaned_tweet_text_pipeline2')}")

# Evaluating results from both pipelines
# with the irish unity dataset
print(f"POST PROCESSING PIPELINE 1:The 20 most commonly used words in tweets\
      mentioning a united Ireland were':\n{get_common_words(sinn_fein_df, 'cleaned_tweet_text_pipeline1')}")

print(f"POST PROCESSING PIPELINE 2:The 20 most commonly used words in tweets\
      mentioning a united Ireland were':\n{get_common_words(sinn_fein_df, 'cleaned_tweet_text_pipeline2')}")


# Output the cleaned dataset to a CSV file
sinn_fein_df.to_csv('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\data_preparation\\sinn_fein_tweets_CLEAN.csv', index=False)
irish_unity_df.to_csv('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\data_preparation\\irish_unity_tweets_CLEAN.csv', index=False)

# Print the first 5 rows of the cleaned dataset
print(sinn_fein_df.head())
print(irish_unity_df.head())