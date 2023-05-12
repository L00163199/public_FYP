# Student Name: Dylan Walsh
# Student ID: L00163199
# Description: The purpose of this file is to use
# the googlesearch library to gain access to the 
# Google search API to mine news headlines over
# the last decade that mention Sinn Fein or Irish Unity
from googlesearch import search
import datetime
import csv
import time

# Mine 20,000 headlines from the last 10
# years that mention Sinn Fein
sinn_fein_query = "Sinn Fein site:news.google.com"
sinn_fein_num_results = 20000
sinn_fein_start_date = datetime.datetime.now() - datetime.timedelta(days=3650)
sinn_fein_stop_date = datetime.datetime.now()

# Mine 20,000 headlines from the last 10
# years that mention Irish Unity OR United Ireland
unity_query = "(Irish Unity OR United Ireland) site:news.google.com"
unity_num_results = 20000
unity_start_date = datetime.datetime.now() - datetime.timedelta(days=3650)
unity_stop_date = datetime.datetime.now()

# Store the headlines
sinn_fein_headlines = []
irish_unity_headlines = []

# Search for Sinn Fein headlines
while len(set(sinn_fein_headlines)) < sinn_fein_num_results:
    
    # 10 results per search to try avoid rate limit
    for url in search(sinn_fein_query, num_results=10):
        if url not in sinn_fein_headlines:
            sinn_fein_headlines.append(url)
        if len(sinn_fein_headlines) >= sinn_fein_num_results:
            break
    # Delay between queries to try avoid
    # being rate limited
    time.sleep(35)

# Search for Irish Unity or United Ireland headlines
while len(set(irish_unity_headlines)) < unity_num_results:
    for url in search(unity_query, num_results=10):
        if url not in irish_unity_headlines:
            irish_unity_headlines.append(url)
        if len(irish_unity_headlines) >= unity_num_results:
            break
    time.sleep(35)

# Concatenate the stored headlines
results = sinn_fein_headlines + irish_unity_headlines
results = list(set(results))

# Generate the results to the file and save
with open('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\data_collection\\headlines.csv', mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Headline', 'Date', 'Type'])
    for url in results:
        writer.writerow([url.split(' - ')[0], url.split(' - ')[1], 'Sinn Fein' if 'sinn-fein' in url.lower() else 'Irish Unity/United Ireland'])