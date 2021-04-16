# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:38:43 2021

@author: micha
"""
## When you run the program search http://127.0.0.1:8050/ and the 
#  interactive dashboard should be visible and working

import dash #  helps you initialize your application (dashboard)
import dash_core_components as dcc # allows you to create interactive components like graphs, dropdowns, or date ranges
import dash_html_components as html # lets you access HTML tags
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

## Read in the 3 datasets. 
datasetJan20 = pd.read_csv("Homelessness Report January 2020.csv")
datasetJune20 = pd.read_csv("Homelessness Report June 2020.csv")
datasetDec20 = pd.read_csv("Homelessness Report December 2020.csv")

### Now add a date column to each dataset
# Date must be formatted in format YYYY-MM-DD in plotly
datasetJan20['date'] = pd.to_datetime("2020-01-01", format='%Y-%m-%d')
#datasetJan20['date'] = datasetJan20['date'].dt.date ## Removes the end of the timestamp leaving just the date YYYY-mm-dd

datasetJune20['date'] = pd.to_datetime("2020-06-01", format='%Y-%m-%d')
#datasetJune20['date'] = datasetJune20['date'].dt.date

datasetDec20['date'] = pd.to_datetime("2020-12-01", format='%Y-%m-%d')
#datasetDec20['date'] = datasetDec20['date'].dt.date

### I will now merge the datasets
# Since the pandas dataframes have matching columns I will append them together
df = datasetJan20.append(datasetJune20.append(datasetDec20))

## Reorder df columns so the date is the first column
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]


## Now begin creating a Dash Dashboard by Plotly
# First will link a CSS stylesheet located in my working directory 
external_stylesheets = [
    {
        "href": "style.css"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
# The above code specifies an external CSS files and a font family. 
# This will be loaded before the body of the application loads and added to 
# the head tag of the application

app = dash.Dash(__name__, external_stylesheets = external_stylesheets) ## creates instance of the dash class
# server = app.server # Tried releasing the app on Heroku but can't get it to work right on that platform
app.title = "Homelessness Dashboard" ## This is the text that will appear in the title bar of your web browser 

# Define the layout of the dash application
app.layout = html.Div(
    children = [
        html.Div(
            children = [
                html.P(children="🛏️", className="header-emoji"),
                html.H1(children="Homelessness Ireland 2020",
                        className="header-title"), 
                html.P(
                    children="This dashboard examines the homelessness"
                    " situation in Ireland over the course of 3 months in 2020:"
                    " January, June, December",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Region", className="menu-title"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {"label":region, "value":region}
                                for region in np.sort(df.Region.unique())
                            ],
                            value="Dublin",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Date Range", className="menu-title"),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=df.date.min().date(),
                            max_date_allowed=df.date.max().date(),
                            start_date=df.date.min().date(),
                            end_date=df.date.max().date(),
                        )
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="Region-chart", config={"displayModeBar" : False},
                    ),
                    className="card"
                ),
                html.Div(
                    children=dcc.Graph(
                        id="Adult-chart", config={"displayModeBar" : False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="Families-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)

#### Dashboard Styling
# Was done by importing a CSS sheet 
# Alternatively it could have been done like so:
#html.H1(
 #    children = "Homelessness 2020",
  #   style = {"fontSize":"48px", "color":"blue"},   
#),
## Doing it the above way is very slow and tedious however, so I used a CSS sheet


#### Making the Dashboard interative
## Using a callback to make the grapgs interactive. The inputs and outputs are clearly defined here. 
## These onjects have two arguments:
    # The indentifier element that they'll modify when the function executes
    # The property of the element to be modified. 
@app.callback(
    [Output("Region-chart", "figure"), Output("Adult-chart", "figure"), Output("Families-chart", "figure")],
    [
        Input("region-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
  # The  Input("region-filter", "value") line will effectively watch the "region-filter" element for changes
  # and will take its value property if the element changes
  
def update_charts(region, start_date, end_date):
    mask = (
        (df.Region == region)
        & (df.date >= start_date)
        & (df.date <= end_date)
    )
    filtered_data = df.loc[mask, :]
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=("", "Breakdown of Homeless Adults by Region", ""), specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])
    
    fig.add_trace(go.Pie( 
            values= df["Total Adults"].head(9), 
            labels=df["Region"].head(9),
            title ="January"
        ),  
        row=1, col=1)
    fig.add_trace(go.Pie( 
            values= df["Total Adults"].head(18).tail(9), 
            labels=df["Region"].head(18).tail(9),
            title ="June"
        ),
        row=1, col=2)
    fig.add_trace(go.Pie( 
            values=df["Total Adults"].tail(9), 
            labels=df["Region"].tail(9),
            title ="December"
        ),
        row=1, col=3)
    
    # region_chart_figure = px.pie(df.head(9), values="Total Adults", names="Region", title= "Regional Breakdown of homeless Adults")
    region_chart_figure = fig
     
    adult_chart_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data["Total Adults"],
                "type": "bar",
                "hovertemplate": "%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Total Homeless Adults",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    family_chart_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data["Number of Families"],
                "type": "bar",
            },
        ],
        "layout": {
            "title": {"text": "Homeless Families", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return region_chart_figure, adult_chart_figure, family_chart_figure






#### Run application
## The following two lines of code run the application locally using Flask’s built-in server.
if __name__ == "__main__":
    app.run_server(debug=True) #  enables the hot-reloading option in your application. 
                               #  This means that when you make a change to your app,
                               #  it reloads automatically, without you having to restart the server.
## When this is run a web page should load. 
