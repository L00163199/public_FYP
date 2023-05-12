"""
Student Name: Dylan Walsh
Student Number: L00163199
Description: The purpose of this file is to
perform exploratory analysis on the datasets
of tweets in order to try and derive meaningful
information that can assist this research

"""
import gensim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from collections import Counter
from prophet import Prophet
from scipy.stats import pearsonr, ttest_ind
from gensim import corpora

# Import the cleaned datasets with sentiment scores/labels
sinn_fein_df = \
    pd.read_csv('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\model_training_evaluation\\sinn_fein_tweets_SENTIMENT.csv')
irish_unity_df = \
    pd.read_csv('C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\model_training_evaluation\\irish_unity_tweets_SENTIMENT.csv')

# Checking the data is as expected
print(sinn_fein_df.head())
print(irish_unity_df.head())

# Merging the datasets for analysis
# This should make it easier than performing
# analysis on each dataset separately, however
# this will only be done where appropriate and
# visually feasible (to avoid information overload)
merged_df = pd.concat([sinn_fein_df, irish_unity_df], keys=['Sinn Fein', 'Irish Unity'])

# Reshape the dataframe for plotting
merged_df = merged_df.reset_index().melt(id_vars=['level_0', 'level_1'],\
                                          value_vars=['sentiment_pipeline1', 'sentiment_pipeline2'],\
                                             var_name='pipeline', value_name='sentiment')


# Generate the boxplot
# Note, the visualisations don't need to
# be aesthetically pleasing at this stage,
# that will be more important for the web app
# when the end user will be seeing them, at this
# level it is just for the developer's own benefit
sns.boxplot(x='level_0', y='sentiment', hue='pipeline',data=merged_df, hue_order=["sentiment_pipeline1", "sentiment_pipeline2"], palette='Set2')

# Setting the labels
plt.xlabel('Dataset')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Scores towards Irish Unity and Sinn Fein')

plt.show()

# Generating a countplot that will visualise
# the number of positive/negative/neutral
# sentiments towards Irish Unity based on
# the sentiment results from pipeline 1
sns.countplot(x='sentiment_label_pipeline1', data=irish_unity_df)
plt.title('Quantity of Sentiment Polarities from Pipeline 1 towards Irish Unity')
plt.show()

# Note - at time of execution pipeline 2
# seems to produce significantly more positive
# sentiment than pipeline 1, this is likely due
# to pipeline 1 using stemming, which can cause
# inaccuracies with words, possibly impacting sentiment
sns.countplot(x='sentiment_label_pipeline2', data=irish_unity_df)
plt.title('Quantity of Sentiment Polarities from Pipeline 2 towards Irish Unity')
plt.show()

# Sinn Fein Dataset
sns.countplot(x='sentiment_label_pipeline1', data=sinn_fein_df)
plt.title('Quantity of Sentiment Polarities from Pipeline 1 towards Sinn Fein')
plt.show()

sns.countplot(x='sentiment_label_pipeline2', data=sinn_fein_df)
plt.title('Quantity of Sentiment Polarities from Pipeline 2 towards Sinn Fein')
plt.show()

# Network Analysis
# Attempting to see the relationships between
# counties and their sentiment towards Sinn Fein
# and Irish unity, may give useful insight into how
# the Northern Irish counties different in sentiment
# in comparison to the republic
unity_county_sentiment = irish_unity_df[['county', 'cleaned_tweet_text_pipeline2', 'sentiment_pipeline2']]
sf_county_sentiment = sinn_fein_df[['county', 'cleaned_tweet_text_pipeline2', 'sentiment_pipeline2']]

# Merging the dataframes
combined_county_df = pd.concat([sf_county_sentiment, unity_county_sentiment])

# Create network graph
network_graph = nx.from_pandas_edgelist(combined_county_df, source='county',\
     target='cleaned_tweet_text_pipeline2')

# Because there can be many nodes generated
# by such a graph, we need to remove those
# of less importance, in this case, dropping
# any nodes that have less than 10 edges, as
# with less edges, there is less impact on the
# other nodes and overall graph
node_degrees = dict(network_graph.degree(network_graph.nodes()))
nx.set_node_attributes(network_graph, node_degrees, 'degree')
relevant_nodes = [n for n in network_graph.nodes() if network_graph.nodes[n]['degree'] > 10]
network_graph = network_graph.subgraph(relevant_nodes)

# The sentiment will be represented as
# the edges and will be given a colour
# representative of such, with a deep
# green being very positive, to a deep
# red being very negative, to a yellow
# being neutral
colour_map = plt.cm.RdYlGn
min_sentiment_score = combined_county_df['sentiment_pipeline2'].min()
max_sentiment_score = combined_county_df['sentiment_pipeline2'].max()
edge_colour = [colour_map((x - min_sentiment_score) / (max_sentiment_score - min_sentiment_score))\
               for x in combined_county_df['sentiment_pipeline2']]

# Visualisation of the graph
node_position = nx.spring_layout(network_graph, k=0.5, iterations=50)
nx.draw_networkx_nodes(network_graph, node_position, node_size=100, node_color='gray')
nx.draw_networkx_edges(network_graph, node_position, edge_color=edge_colour, width=2)
nx.draw_networkx_labels(network_graph, node_position, font_size=8, font_family='arial')

# Adding colour bar legend to visualise
# the range of colour representing the range
# of sentiment polarity, from positive to negative
sentiment_colour_map = plt.cm.ScalarMappable(cmap=colour_map,\
                           norm=plt.Normalize(vmin=min_sentiment_score,\
                                               vmax=max_sentiment_score))
sentiment_colour_map._A = []
colour_bar = plt.colorbar(sentiment_colour_map)
colour_bar.ax.set_ylabel('Sentiment Score', rotation=0, labelpad=16)

# Labelling the graph
plt.title('Sentiment Towards Irish Unity and Sinn Fein by County based on Tweet Text')
plt.xlabel('County')
plt.ylabel('Tweet Text')
plt.show()

# Generating cloud based on Sinn Fein text
# Note - at this stage from execution it has
# been determined the first preprocessing/cleaning
# pipeline that incorporates stemming is not the
# best for accuracy, therefore pipeline 2
# will be used from here onwards
sinn_fein_text = ' '.join(sinn_fein_df['cleaned_tweet_text_pipeline2'].astype(str).tolist())
sf_word_count = Counter(sinn_fein_text.split())
sf_word_cloud = WordCloud(init_opts=opts.InitOpts(width="1000px", height="1000px"))
sf_word_cloud.add("", list(sf_word_count.items()), word_size_range=[20, 100], shape="circle")
sf_word_cloud.set_global_opts(title_opts=opts.TitleOpts(title="Sinn Fein Tweets Word Cloud"))

# Rendering word cloud
sf_word_cloud.render("C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\model_training_evaluation\\sinn_fein_wordcloud.html")

# Generating cloud based on Irish Unity text
irish_unity_text = ' '.join(irish_unity_df['cleaned_tweet_text_pipeline2'].astype(str).tolist())
irish_unity_word_counts = Counter(irish_unity_text.split())
irish_unity_wordcloud = WordCloud(init_opts=opts.InitOpts(width="800px", height="800px"))
irish_unity_wordcloud.add("", list(irish_unity_word_counts.items()), word_size_range=[20, 100], shape="circle")
irish_unity_wordcloud.set_global_opts(title_opts=opts.TitleOpts(title="Word Cloud of Irish Unity Tweets"))

# Rendering word cloud
irish_unity_wordcloud.render("C:\\Users\\Dyl\\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\model_training_evaluation\\irish_unity_wordcloud.html")


# Ensure text is the preprocessed text based
# on pipeline to and is of type string to avoid
# any issues with data type
sinn_fein_texts = sinn_fein_df['cleaned_tweet_text_pipeline2'].astype(str).tolist()
sf_cleaned_text = [text.split() for text in sinn_fein_texts]

# Creating dictionary of unique words from
# the cleaned/preprocessed sinn fein text
# to train the LDA model
sinn_fein_dictionary = corpora.Dictionary(sf_cleaned_text)
sf_unique_words = [sinn_fein_dictionary.doc2bow(text) for text in sf_cleaned_text]

# Training the LDA model
# Note at time of execution on this machine
# it takes about 2 minutes
sinn_fein_lda_model = gensim.models.ldamodel.LdaModel(corpus=sf_unique_words,
                                                      id2word=sinn_fein_dictionary,
                                                      num_topics=10,
                                                      random_state=42,
                                                      passes=10)

# Output the most popular topics for Sinn Fein
print('The most discussed topics regarding Sinn Fein were:')
for sf_topic in sinn_fein_lda_model.print_topics():
    print(sf_topic)

# Repeating the process but for Irish Unity
irish_unity_texts = irish_unity_df['cleaned_tweet_text_pipeline2'].astype(str).tolist()
unity_cleaned_text = [text.split() for text in irish_unity_texts]

irish_unity_dictionary = corpora.Dictionary(unity_cleaned_text)
irish_unity_corpus = [irish_unity_dictionary.doc2bow(text) for text in unity_cleaned_text]

# Training the LDA model
irish_unity_lda_model = gensim.models.ldamodel.LdaModel(corpus=irish_unity_corpus,
                                                        id2word=irish_unity_dictionary,
                                                        num_topics=5,
                                                        random_state=42,
                                                        passes=10)

# Output the most popular topics for Irish Unity
print('The most discussed topics regarding Irish Unity were:')
for unity_topic in irish_unity_lda_model.print_topics():
    print(unity_topic)

# Time Series Analysis
# Viewing the annual quantity of sentiment
# polarities based on Irish Unity and Sinn Fein

# Converting to datetime format as is
# currently in string format from the dataset
irish_unity_df['date_posted'] = pd.to_datetime(irish_unity_df['date_posted'], format='%d/%m/%Y')
sinn_fein_df['date_posted'] = pd.to_datetime(sinn_fein_df['date_posted'], format='%d/%m/%Y')

# Group the cleaned tweets by year and sentiment label
# then count the number of tweets annually
# Due to grouping, the data has a more complex
# structure with multiple index levels, this
# can be made easier for visualisation by 
# resetting the index
unity_tweets_yearly_sentiment = \
    irish_unity_df.groupby([irish_unity_df['date_posted'].dt.year,\
                             'sentiment_label_pipeline2']).size().reset_index(name='count')
sf_tweets_yearly_sentiment = \
    sinn_fein_df.groupby([sinn_fein_df['date_posted'].dt.year, \
                          'sentiment_label_pipeline2']).size().reset_index(name='count')

# Preparing data for visualisation
# by reshaping it with pivot tables
unity_tweets_yearly_sentiment_pivot = \
    unity_tweets_yearly_sentiment.pivot(index='date_posted',\
                                                columns='sentiment_label_pipeline2', values='count')
sf_tweets_yearly_sentiment_pivot = \
    sf_tweets_yearly_sentiment.pivot(index='date_posted', \
                                           columns='sentiment_label_pipeline2', values='count')

# Generating the time series visualisation
unity_tweets_yearly_sentiment_pivot.plot()
plt.xlabel('Year')
plt.ylabel('Number of Tweets')
plt.title('Annual Quantity of Sentiment towards Irish Unity')
plt.show()

sf_tweets_yearly_sentiment_pivot.plot()
plt.xlabel('Year')
plt.ylabel('Number of Tweets')
plt.title('Annual Quantity of Sentiment towards Sinn Fein')
plt.show()

# Pearson's coefficient
# Using to determine if there is a 
# relationship between the annual sentiment
# towards Sinn Fein and Irish Unity

# Generate the new column by deriving the
# year from the date_posted column
irish_unity_df['year'] = irish_unity_df['date_posted'].dt.year
sinn_fein_df['year'] = sinn_fein_df['date_posted'].dt.year

# Getting the mean annual sentiment
irish_unity_sentiment = irish_unity_df.groupby(['year'])['sentiment_pipeline2'].mean().reset_index()
sinn_fein_sentiment = sinn_fein_df.groupby(['year'])['sentiment_pipeline2'].mean().reset_index()

# Merging the datasets based on the year
merged_sentiments = pd.merge(irish_unity_sentiment, sinn_fein_sentiment, on='year')

# Attempting to determine the correlation
# between the annual sentiment towards Sinn Fein
# and Irish Unity using pearson's correlation coefficient
correlation, pvalue = pearsonr(merged_sentiments['sentiment_pipeline2_x'],\
                                merged_sentiments['sentiment_pipeline2_y'])
print(f"Correlation between Sinn Fein and Irish Unity based on annual sentiment: {correlation:.2f}")

# Visualising the correlation
plt.figure(figsize=(10,6))
plt.plot(merged_sentiments['year'], merged_sentiments['sentiment_pipeline2_x'], label='Annual Sentiment for Irish Unity')
plt.plot(merged_sentiments['year'], merged_sentiments['sentiment_pipeline2_y'], label='Annual Sentiment for Sinn Fein')
plt.xlabel('Year')
plt.ylabel('Mean Sentiment Score')
plt.title('Mean Annual Sentiment Scores for Irish Unity and Sinn Fein')
plt.legend()
plt.show()

# Time Series Forecast
# Implementing predictive sentiment analysis
# through the use of time series forecasting
# of annual sentiment towards Sinn Fein and
# Irish Unity based on the last decades worth
# of tweets from the datasets, also trying to
# determine if there is any relationship between
# the annual sentiment towards Sinn Fein/Irish Unity

# Creating a dataframe to hold the date posted
# and the sentiment scores from pipeline 2
irish_unity_df = pd.DataFrame({'ds': irish_unity_df['date_posted'], 'y': irish_unity_df['sentiment_pipeline2']})
sinn_fein_df = pd.DataFrame({'ds': sinn_fein_df['date_posted'], 'y': sinn_fein_df['sentiment_pipeline2']})

# Creating the prophet model for forecasting
# Irish Unity sentiment
irish_unity_prophet = Prophet()
irish_unity_prophet.fit(irish_unity_df)

sinn_fein_prophet = Prophet()
sinn_fein_prophet.fit(sinn_fein_df)

# Creating a new dataframe to hold the
# data for the next decade
next_decade_sentiment = pd.DataFrame({'ds': pd.date_range(start='2023-01-01', end='2032-12-31', freq='AS')})

# Implementing the forecasting
irish_unity_forecast = irish_unity_prophet.predict(next_decade_sentiment)
sinn_fein_forecast = sinn_fein_prophet.predict(next_decade_sentiment)

# Visualising the forecast
plt.figure(figsize=(10, 5))
plt.plot(irish_unity_forecast['ds'], irish_unity_forecast['yhat'], label='Irish Unity')
plt.plot(sinn_fein_forecast['ds'], sinn_fein_forecast['yhat'], label='Sinn Fein')
plt.legend(loc='upper left')
plt.title('Mean Annual Sentiment Forecast for Irish Unity and Sinn Fein over next Decade')
plt.xlabel('Year')
plt.ylabel('Sentiment Score')
plt.show()