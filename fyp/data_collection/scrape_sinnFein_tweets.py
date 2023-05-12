"""
Student Name: Dylan Walsh
Student Number: L00163199
Description: The purpose of this file is to mine
a sum of tweets from Twitter that are discussing
the Irish nationalist party Sinn Fein and to extract
certain demographic information about the posters of these
tweets

"""

# Note - Due to the Twitter API being heavily
# restricted for several reasons, the web scraping 
# tool snscrape will be used to get past this. Any
# data collected will only be used for the purposes 
# of this project.

# The snscrape library is installed locally on
# this machine but to install on your machine
# you can use the following: pip install snscrape

# Widely used in data science, useful in its ability
# to handle large sets of data, have used several times
# in previous modules such as AI & Machine Learning
import pandas as pd
import itertools
import re

# This package was reviewed from the following link:
# https://github.com/JustAnotherArchivist/snscrape
import snscrape.modules.twitter as twitterscraper

# Since this research focuses on investigating
# the likelihood of Irish unification within
# the next decade, it is reasonable to say that
# scraping tweets from the last decade would be
# of benefit for analysis
num_of_tweets = 200000
from_date = "2013-01-30"
to_date = "2023-01-30"

# Snscrape is case-sensitive, we can use the 
# OR clause to widen the tweets scraped
search_query = 'sinn fein OR Sinn Fein OR SINN FEIN OR #sinnfein'

# Using regex patterns to identify demographic info
# This is not an ideal solution - it can miss
# information if it doesn't match the pattern
# and there is no way to validate the context 
# of the matched patterns either, should be
# considered as synthetic data

# Inspiration for this was derived from the following:
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262087
age_pattern = re.compile(r'(?:I\W+am\W+|\bi\W+was\W+born\W+on\W+|\bi\W+am\W+)(\d{1,3})\b', re.IGNORECASE)
county_pattern = re.compile(r'\b(co\.|county)\s+\w+\b\s+\w+', re.IGNORECASE)
religion_pattern = re.compile(r'\b(am\s)?(catholic|protestant|atheist)\b', re.IGNORECASE)

# Finding the user's profile based on username
def get_user_profile(username):
    try:
        user_scraper = twitterscraper.TwitterUserScraper(username)
        return user_scraper.get_user()
    except Exception as profile_exception:
        print(f"Could not find user profile: {profile_exception}")
        return None

# Extracts specific information such as
# a user's age, county they are from and
# religion using regex
def get_user_demographic(tweet):

    # If the contents of the tweet mention sinn fein
    # then go to the user's profile
    if 'sinn fein' in tweet.tweet_content.lower():

        # Pulling the user's profile bio
        twitter_user = get_user_profile(tweet.posted_by)
        if twitter_user is not None:
            bio_content = twitter_user.bio
            user_location = twitter_user.location
            user_age = age_pattern.search(bio_content)
            county_match = county_pattern.search(user_location)
            religion_match = religion_pattern.search(bio_content)

            age = user_age.group(0) if user_age else None
            county = county_match.group(0) if county_match else None
            religion = religion_match.group(0) if religion_match else None

        # returns a dictionary containing the
        # extracted demographic info
        return {
            'age': age,
            'county': county,
            'religion': religion
        }
    else:
        return None
    
# Performs a search query to scrape tweets 
# based on some given number of parameters
def get_tweets(search_query, num_of_tweets, from_date, to_date):
    scraped_tweets = []

    try:
        tweet_scraper = twitterscraper.TwitterSearchScraper(f'{search_query}\
                        since:{from_date} until:{to_date}')
        
        # Call islice function to give more control
        # over how many tweets are returns in an
        # iterator rather than iterating over more
        # tweets than necessary
        for tweet in itertools.islice(tweet_scraper.get_items(),num_of_tweets):
            tweet_info = {
                'date_posted': tweet.date,
                'tweet_id': tweet.id,
                'tweet_text': tweet.content,
                'user_location': tweet.user.location,
                'twitter_user': tweet.user.username,
            }
            demographic_info = get_user_demographic(tweet)

            # Update the dictionary with the demographic info
            tweet_info.update(demographic_info)
            scraped_tweets.append(tweet_info)
            
    except ValueError as val_error:
        print(f"An invalid value was passed: {val_error}")
    # Should only occur if issue with date format
    except KeyError as key_error:
        print(f"An invalid date range was provided: {key_error}")
    except TypeError as type_error:
        print(f"An invalid type was passed: {type_error}")
    
    for tweet in scraped_tweets:
        demographic_info = get_user_demographic(tweet)
        if demographic_info is not None:
            tweet.update(demographic_info)

            # Now that the demographic info has been extracted
            # the tweet ID and username fields can be dropped
            # in order to avoid potential ethical issues with
            # privacy/informed consent, by removing them we 
            # can anonymize the extracted data and fall in line
            # with GDPR guidelines on anonymisation
            tweet.pop('twitter_user', None)
            tweet.pop('tweet_id')

            # Not needed as county is extracted now
            tweet.pop('user_location')
        
    return scraped_tweets

# Note - At time of execution, this machine took 
# 1 hour and 10 minutes in total to finish mining
scraped_tweets = get_tweets(search_query, num_of_tweets, from_date, to_date)
df = pd.DataFrame(scraped_tweets)
df.to_csv('sinn_fein_tweets_RAW.csv', index=False)





