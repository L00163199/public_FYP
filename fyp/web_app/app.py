# Student Name: Dylan Walsh
# Student Number: L00163199
# Description: The purpose of this file is to
# contain all functions required to reproduce
# the visualisations as per the first half
# of this project into a more interactive and
# accessible format for the end user

from flask import Flask, render_template, request, redirect, url_for, session, flash, url_for, make_response
from werkzeug.utils import secure_filename
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pyecharts.charts import WordCloud
from flask import jsonify
from pyecharts import options as opts
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from dateutil.relativedelta import relativedelta
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
from prophet import Prophet
from matplotlib import patches
from plotly.tools import mpl_to_plotly
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import plotly.utils
import statsmodels.api as sm
import os
import math
import plotly
import pandas as pd
import io
import base64
import json
import textwrap
import seaborn as sns
import gensim
import networkx as nx
import matplotlib as plt
import textwrap

register_matplotlib_converters()

# Refers to the html files
app = Flask(__name__, template_folder='templates')

# Temporary folder to store uploaded files
# by the end user and to also store generated
# visualisations where appropriate
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\Dyl\OneDrive - Atlantic TU\\semester_6\\project_dev\\fyp\\web_app\\uploads'

# Randomly created key for this project
app.secret_key = 'atudcs23fypdw'

# Only want to allow .csv files
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# The first thing that should be done is
# uploading the file, therefore display
# the upload page on first run
@app.route('/')
def initial():
    return render_template('upload.html')

# Index is the base layout for the web app
# Once the file is uploaded it will be displayed
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            file.save(filename)
            flash("File uploaded successfully!")
            return redirect(url_for("index"))

    return render_template("index.html")

# Routes for the navigation menu options
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/datasets')
def datasets():
    return render_template('datasets.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Function to check if file has
# the .csv extension on it
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):

            # Save the file locally to the uploads folder
            # Once saved the dataset can be accessed
            # to use for the other functionalities later
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            # Store file path in session
            session['file_path'] = file_path
            # Check if file exists
            if os.path.exists(file_path):
                message = 'File uploaded successfully!'
                flash(message)
                return redirect(url_for('index'))
            else:
                message = 'Upload failed, please try again'
                flash(message)
                return redirect(url_for('initial'))
        else:
            message = 'Please ensure the file is of .csv format'
            flash(message)
            return redirect(url_for('initial'))

    return render_template('index.html')

# Creates a bar chart based on
# passed x and y values
def create_bar_chart(df, x_value, y_value, y_range=None):
    data = [{'x': df[x_value], 'y': df[y_value], 'type': 'bar'}]
    layout = {'title': f'{y_value} vs {x_value}', 'xaxis': {'title': x_value}, 'yaxis': {'title': y_value}}
    
    # Sometimes the range of values is done in
    # automatic increments based on passed data,
    # in order for it to be exactly the same as
    # it is in the data, we can specify the range
    # based on the passed variable
    if y_range is not None:
        layout['yaxis']['range'] = y_range
    
    fig = go.Figure(data=data, layout=layout)
    return fig.to_json()

# Creates a box plot similar to
# how the bar chart is made
def create_box_plot(df, x_value, y_value, y_range=None):
    data = [{'x': df[x_value], 'y': df[y_value], 'type': 'box'}]
    layout = {'title': 'Boxplot of ' + f'{y_value} vs {x_value}','xaxis': {'title': x_value}, 'yaxis': {'title': y_value}}

    if y_range is not None:
        layout['yaxis']['range'] = y_range
    
    fig = go.Figure(data=data, layout=layout)
    return fig.to_json()

@app.route('/barcharts', methods=['GET', 'POST'])
def barcharts():
    if 'file_path' not in session:
        flash('Please upload a file first!', 'error')
        return redirect(url_for('initial'))
    
    # load the dataset
    file_path = session['file_path']
    df = pd.read_csv(file_path)
    
    # get column names for dropdown options
    options = [{'label': col, 'value': col} for col in df.columns]
    
    # Only need to do this is the form
    # has been submitted with the chosen options
    if request.method == 'POST':
        x_value = request.form['x_value']
        y_value = request.form['y_value']
        y_range = [df[y_value].min(), df[y_value].max()]
        
        # create bar chart data
        graphJSON = create_bar_chart(df, x_value, y_value, y_range=y_range)
        
        return render_template('barcharts.html', graphJSON=graphJSON, options=options, x_value=x_value, y_value=y_value)
      
    return render_template('barcharts.html', options=options)

# Returns a box plot chart
@app.route('/boxplots', methods=['GET', 'POST'])
def boxplots():
    if 'file_path' not in session:
        flash('Please upload a file first!', 'error')
        return redirect(url_for('initial'))

    # load the dataset
    file_path = session['file_path']
    df = pd.read_csv(file_path)

    # get column names for dropdown options
    options = list(df.columns)

    if request.method == 'POST':
        # get selected values
        x_column = request.form['x_column']
        y_column = request.form['y_column']

        # create box plot
        fig_json = create_box_plot(df, x_column, y_column)

        return render_template('boxplots.html', options=options, x_column=x_column, y_column=y_column, graphJSON=fig_json)

    return render_template('boxplots.html', options=options)

# Returns a wordcloud visualisation
@app.route('/wordcloud', methods=['GET', 'POST'])
def wordcloud():
    file_path = session['file_path']
    df = pd.read_csv(file_path)
    options = list(df.columns)
    selected_column = options[0]
    
    if request.method == 'POST':
        # Get selected column name from the form
        selected_column = request.form['selected_column']
        
        # Store text from the chosen column
        text = " ".join(df[selected_column].astype(str))
        
        # The chosen column may contain entire strings
        # of text, therefore to create an appropriate
        # wordcloud the strings need to be split into
        # individual words
        words = text.split()

        # Count the word frequency to
        # determine its popularity
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

        # Create a list of tuples containing
        # the word and its associated frequency
        word_freq_list = []
        for word, freq in word_freq.items():
            word_freq_list.append((word, freq))

        # Generate word cloud
        wordcloud = (
            WordCloud()
            .add(series_name=selected_column, data_pair=word_freq_list)
        )
        
        return render_template('wordcloud.html', options=options, selected_column=selected_column, chart=wordcloud.render_embed())

    else:
        return render_template('wordcloud.html', options=options, selected_column=selected_column)

# Creates a network graph based on
# some chosen nodes and edge weights
# from the end user, then saves the
# graph in image data and returns to
# the page
@app.route('/network-graph', methods=['GET', 'POST'])
def network_graph():
    file_path = session['file_path']
    df = pd.read_csv(file_path)
    options = list(df.columns)

    # If form submitted, create network graph
    if request.method == 'POST':
        # Store the chosen variables from the user

        # Chosen nodes and edges
        source_col = request.form['source']
        target_col = request.form['target']
        edge_col = request.form['edge']

        # Chosen nodes, edges, background colors
        source_color = request.form['source-color']
        target_color = request.form['target-color']
        edge_color = request.form['edge-color']
        label_color = request.form['label-color']
        background_color = request.form['bg-color']

        # Create a title for the graph based on nodes/edges
        # Easier to do this as it caters to any dataset
        # rather than the one specific to this project
        title = f"Network Graph for {source_col}, {target_col}, and {edge_col}"

        # Draw the edges from source to target with the edge weights
        edges = []
        for _, row in df[[source_col, target_col, edge_col]].iterrows():
            edges.append((row[source_col], row[target_col], row[edge_col]))

        # Create network graph
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)

        # Only keep nodes with at least 10 edges
        # otherwise there will be too many nodes
        # which will make the graph too illegible
        degree_dict = dict(G.degree(G.nodes()))
        nodes_to_remove = [n for n in degree_dict if degree_dict[n] < 10]
        G.remove_nodes_from(nodes_to_remove)

        # Draw network graph
        fig = Figure(figsize=(12,10))
        canvas = FigureCanvas(fig)

        # To increase space between nodes,
        # increase the value of k
        pos = nx.spring_layout(G, k=3)

        # Defining node colors
        # Note - sometimes when trying to take
        # hex values in string form there can be
        # issues with generating colors, good idea
        # to add them to add dictionary so the hex values
        # can be accessed
        node_colors = []
        color_dict = {source_color: f"Source Nodes ({source_col})", target_color: f"Target Nodes ({target_col})", edge_color: f"Edges (values are in the white box): ({edge_col})"}
        
        for node in G.nodes():
            if node in df[source_col].values:
                node_colors.append(source_color)
            elif node in df[target_col].values:
                node_colors.append(target_color)
            else:
                node_colors.append('grey')

        # Define edge colors
        edge_colors = [edge_color] * len(G.edges())

        # Add the colors to the nodes and edges
        ax = fig.gca()
        ax.set_facecolor(background_color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax)

        # Generates a legend for inside the graph
        def draw_legend(ax, color_dict):
            handles = []
            labels = []
            for color, label in color_dict.items():
                handles.append(patches.Patch(color=color))
                labels.append(label)

            # Note, need to use bbox_to_anchor to move
            # the legend to the end of the box so it
            # doesn't overlap with the graph
            ax.legend(handles, labels, bbox_to_anchor=(1.01, 0.06), loc='upper right', fontsize=10, frameon=True, framealpha=1.0, facecolor='white', edgecolor='black', title="Legend")

        draw_legend(ax, color_dict)
        
        # With this project specifically, when the 
        # target node is the tweet text, it can be
        # very difficult to read on the graph as some
        # strings of text are too long, this can be avoided
        # by forcing the text into a box and breaking it
        # into multiple lines to make it easier to read
        node_labels = {}
        for node in G.nodes():
            if node in df[target_col].values:
                node_labels[node] = node
                x, y = pos[node]
                text = node_labels[node]
                width = len(text)*0.02
                height = 0.06
                box_style = dict(facecolor='white', edgecolor=target_color, pad=0.2)

                # Note: The box will likely cover the node shape
                # so to indicate it is the target node, set the border
                # to the same color as the target node
                ax.text(x, y, '\n'.join(textwrap.wrap(text, width=10)), ha='center', va='center', color=label_color, fontsize=10, fontweight='bold', zorder=5, bbox=box_style)
                ax.add_patch(patches.Rectangle((x-width/2,y-height/2),width,height,facecolor='none',edgecolor='none',zorder=4))

                
        # Draw source node labels
        source_node_labels = {}
        for node in G.nodes():
            if node in df[source_col].values:
                source_node_labels[node] = node
        nx.draw_networkx_labels(G, pos, labels=source_node_labels, font_color=label_color, ax=ax)

        # Display the edge weights
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_color='black')

        # Save the image in the temporary folder
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'network_graph.png')
        fig.savefig(img_path)
        with open(img_path, 'rb') as f:
            img_data = f.read()

        # Return the image data
        response = make_response(img_data)
        response.headers.set('Content-Type', 'image/png')
        return response

    # Otherwise return the html file with the column options
    return render_template('network_diagram.html', options=options)


# Performs pearson's coefficient correlation
# and returns a visualisation of it
@app.route('/pearson', methods=['GET', 'POST'])
def pearson():
    if 'file_path' not in session:
        flash('Please upload a file first!', 'error')
        return redirect(url_for('index'))

    file_path = session['file_path']
    df = pd.read_csv(file_path)
    options = list(df.columns)

    if request.method == 'POST':
        x_value = request.form['x_value']
        y_value = request.form['y_value']

        corr = df[x_value].corr(df[y_value])

        # 2 numeric columns must be chosen
        if pd.isna(corr):
            flash('Invalid input. Please select two columns with valid numeric values.', 'error')
            return redirect(url_for('pearson'))

        # Using a scatter plot to show the correlation
        trace = go.Scatter(x=df[x_value], y=df[y_value], mode='markers')

        # Further adding to the visualisation by including the percentage
        layout = go.Layout(title=f'Pearson Correlation Coefficient: {corr:.2f}',
                           xaxis_title=x_value, yaxis_title=y_value)
        
        fig = go.Figure(data=[trace], layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('correlation.html', options=options, graphJSON=graphJSON,
                               x_value=x_value, y_value=y_value, corr=corr)

    return render_template('correlation.html', options=options)


# Performs and returns a visualisation
# of a time series analysis
@app.route('/time_series_analysis', methods=['GET', 'POST'])
def time_series_analysis():
    if 'file_path' not in session:
        flash('Please upload a file first!', 'error')
        return redirect(url_for('index'))

    file_path = session['file_path']
    df = pd.read_csv(file_path)
    options = list(df.columns)

    if request.method == 'POST':
        x_value = request.form['x_value']
        y_value = request.form['y_value']

        # This is specific to the dataset for this project
        # It will group the column of dates into years and
        # get the mean annual value of the y column
        df[x_value] = pd.to_datetime(df[x_value], format='%d/%m/%Y')
        min_year = df[x_value].min().year
        max_year = df[x_value].max().year
        annual_mean_df = df.groupby(df[x_value].dt.year)[y_value].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=annual_mean_df.index, y=annual_mean_df.values))
        fig.update_layout(title=f'Time Series Plot for {y_value}', xaxis_title='Year', yaxis_title=y_value, template='plotly_white')
        graphJSON = fig.to_json()

        return render_template('time_series_analysis.html', options=options, graphJSON=graphJSON,
                               x_value=x_value, y_value=y_value)

    return render_template('time_series_analysis.html', options=options)


# Performs and returns a visualisation
# of a time series forecast
@app.route('/time_series_forecast', methods=['GET', 'POST'])
def time_series_forecast():
    if 'file_path' not in session:
        flash('Please upload a file first!', 'error')
        return redirect(url_for('index'))

    file_path = session['file_path']
    df = pd.read_csv(file_path)
    options = list(df.columns)

    if request.method == 'POST':
        x_value = request.form['x_value']
        y_value = request.form['y_value']
        
        df[x_value] = pd.to_datetime(df[x_value], format='%d/%m/%Y')

        # The way this forecast was approached is to
        # get the difference between the earliest and 
        # latest date in years and then use this difference
        # to determine the length of the forecast
        # For example, if the range in dates in the dataset
        # was 10 years, it will make a forecast for the 
        # next 10 years
        min_date = df[x_value].min()
        max_date = df[x_value].max()
        date_diff = relativedelta(max_date, min_date)
        date_range_years = date_diff.years + date_diff.months / 12 + date_diff.days / 365
        date_range_years = int(date_range_years)

        df.rename(columns={x_value: 'ds', y_value: 'y'}, inplace=True)

        # Using prophet to perform the forecast
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=date_range_years*12, freq='M')
        forecast = m.predict(future)

        # Plot the forecasted values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
        fig.update_layout(title=f'Time Series Forecast for {y_value}', xaxis_title='Year', yaxis_title=y_value)

        # Since this is a forecast, the start date will
        # be the latest date in the dataset and the end date
        # will be the latest date added with the difference in years
        # For example, 01/01/2013 to 01/01/2023, the difference
        # is 10 years, the start date will be 01/01/2023
        # and therefore the end date will be 01/01/2033
        fig.update_layout(xaxis_range=[max_date, max_date + pd.Timedelta(days=365) + relativedelta(years=date_range_years)])

        # Serialise plotly figure to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('time_series_forecast.html', options=options, graphJSON=graphJSON,
                            x_value=x_value, y_value=y_value)
    
    return render_template('time_series_forecast.html', options=options)


# Run the web app
if __name__ == '__main__':
    app.run(debug=True,port=4000)
