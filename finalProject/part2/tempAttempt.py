# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:36:19 2021

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
import branca.colormap as cm
import folium
import geopandas as gpd


## Read in the 3 datasets. 
datasetJan20 = pd.read_csv("Homelessness Report January 2020.csv")
datasetJune20 = pd.read_csv("Homelessness Report June 2020.csv")

datasetDec20 = pd.read_csv("Homelessness Report December 2020.csv")

### Now add a date column to each dataset
datasetJan20['date'] = "January"
# Date must be formatted in format YYYY-MM-DD in plotly
#datasetJan20['date'] = pd.to_datetime("2020-01-01", format='%Y-%m-%d')
#datasetJan20['date'] = datasetJan20['date'].dt.date ## Removes the end of the timestamp leaving just the date YYYY-mm-dd

datasetJune20['date'] = "June"
#datasetJune20['date'] = pd.to_datetime("2020-06-01", format='%Y-%m-%d')
#datasetJune20['date'] = datasetJune20['date'].dt.date

datasetDec20['date'] = "December"
#datasetDec20['date'] = pd.to_datetime("2020-12-01", format='%Y-%m-%d')
#datasetDec20['date'] = datasetDec20['date'].dt.date

### I will now merge the datasets
# Since the pandas dataframes have matching columns I will append them together
df = datasetJan20.append(datasetJune20.append(datasetDec20))

## Reorder df columns so the date is the first column
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]


#countries = gpd.read_file('python/Countries_WGS84.shp') # Part of my attempt to make a heatmap of Ireland


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
        # First 2 Filters:
        # (When the user uses these filters the graphs that can be filtered by this data will be changed,
        #  code for this is in an update function near the end of the script.)
        # Note: Had to be very careful with filters as for example it is not possible to know how many 
        # females from Dublin accessed private emergency accomodation
        html.Div(
            children=[
                html.Div( 
                    # Region of Ireland Filter
                    children=[
                        html.Div(children="Region", className="menu-title"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {"label":region, "value":region}
                                for region in np.sort(df.Region.unique())
                            ],
                            value=["Dublin", "Mid-East", "Midlands", "Mid-West", "North-East", 
                                   "North-West", "South-East", "South-West", "West"     
                            ], ## Default Values. 
                            clearable=False,
                            multi=True,
                            className="dropdown",
                        ),
                    ]
                ),
                 html.Div(
                    # Gender Filter
                    children=[
                        html.Div(children="Gender", className="menu-title"),
                        dcc.Dropdown(
                            id="gender-filter",
                            options=[
                                {'label':'Male', 'value' : 'Male'},
                                {'label':'Female', 'value' : 'Female'},
                            ],
                            value=["Male","Female"], ## Default (initial) Value. 
                            clearable=False,
                            multi=True,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    # Age selector
                    children=[
                        html.Div(children="Age Range", className="menu-title"),
                        dcc.Dropdown(
                            id="age-filter",
                            options=[
                                {'label':'18-24', 'value' : 'Adults Aged 18-24'},
                                {'label':'25-44', 'value' : 'Adults Aged 25-44'},
                                {'label':'45-64', 'value' : 'Adults Aged 45-64'},
                                {'label':'65+', 'value' : 'Adults Aged 65+'},
                            ],
                            value=["Adults Aged 18-24","Adults Aged 25-44", "Adults Aged 45-64", "Adults Aged 65+"], ## Default (initial) Value. 
                            clearable=False,
                            multi=True,
                            className="dropdown",
                        ),
                    ]
               ),
            ],
            className="first-menu",
        ),
        html.Div(
            children=[
                ####################### Fig 1. The region chart. 
                html.Div(
                    children=dcc.Graph(
                        id="Region-chart", config={"displayModeBar" : False},
                    ),
                    className="card"
                ),
                ########################## Second row of filters
                html.Div(
                    children=[
                        html.Div(
                            # Single Region of Ireland Filter
                            children=[
                                html.Div(children="Region", className="menu-title"),
                                dcc.Dropdown(
                                    id="single-region-filter",
                                    options=[
                                        {"label":region, "value":region}
                                        for region in np.sort(df.Region.unique())
                                    ],
                                    value="Dublin", ## Default Value. 
                                    clearable=False,
                                    className="dropdown",
                                ),
                            ]
                        ),
                        html.Div(
                            # Date selector filter. This was orginally a calander filter, but as there is only
                            # 3 dates (the start of the months Jan, June & Dec) it makes more sense to have a dropdown bar and change the
                            # dates into just the names of the months
                            children=[
                                html.Div(children="Month Selector", className="menu-title"),
                                dcc.Dropdown(
                                    id="month-filter",
                                    options=[
                                        {"label":month, "value":month}
                                        for month in df.date.unique()
                                    ],
                                    value=["January","June", "December"], ## Default (initial) Value. 
                                    clearable=False,
                                    multi=True,
                                    className="dropdown",
                                ),
                            ]
                        ),
                        html.Div(
                            # Age selector
                            children=[
                                html.Div(children="Age Range", className="menu-title"),
                                dcc.Dropdown(
                                    id="age-filter2",
                                    options=[
                                        {'label':'18-24', 'value' : 'Adults Aged 18-24'},
                                        {'label':'25-44', 'value' : 'Adults Aged 25-44'},
                                        {'label':'45-64', 'value' : 'Adults Aged 45-64'},
                                        {'label':'65+', 'value' : 'Adults Aged 65+'},
                                    ],
                                    value=["Adults Aged 18-24","Adults Aged 25-44", "Adults Aged 45-64", "Adults Aged 65+"], ## Default (initial) Value. 
                                    clearable=False,
                                    multi=True, 
                                    className="dropdown",
                                ),
                            ]
                       ),
                    ],
                    className="second-menu",
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
                html.Div(
                   children = [ 
                       html.Div(
                           html.H1("Homeless Statistics Ireland 2020", style={'color':'black', 'padding-top': '10px', 'padding-bottom': '20px'}, className = "header-title" )
                       ),
                       html.Div(html.Img(src="/assets/homelessImage.jfif"), 
                                style={'display': 'inline-block', 'float':'right', 'padding-top': '20px'},
                                className="card",
                       ),
                       html.Div(children=[ 
                                   html.P("Jan 2020", style={'font-size' : '40px'}),
                                   html.Ul(id='my-list', children=[
                                                               html.Li( str(df["Total Adults"].head(9).sum()) + 
                                                                        " Adults"),
                                                               html.Li( str(df["Number of Families"].head(9).sum()) +
                                                                        " Families" ),                                              
                                                         ])
                               ],
                               style={'display': 'inline-block', 'padding-left': '20px', 'padding-right': '30px', 'font-size' : '20px'},
                               className="card",
                       ),
                       html.Div(children=[
                                    html.P("June 2020", style={'font-size' : '40px'}),
                                    html.Ul(id='June-list', children=[
                                                                html.Li( str(df["Total Adults"].head(18).tail(9).sum()) + 
                                                                         " Adults"),
                                                                html.Li( str(df["Number of Families"].head(18).tail(9).sum()) +
                                                                         " Families" ),       
                                                           ])
                                ], 
                                style={'display': 'inline-block', 'padding-left': '20px', 'padding-right': '30px', 'font-size' : '20px'},
                                className="card",
                       ),
                       html.Div(children=[
                                    html.P("Dec 2020", style={'font-size' : '40px'}),
                                    html.Ul(id='Dec-list', children=[
                                                                html.Li( str(df["Total Adults"].tail(9).sum()) + 
                                                                         " Adults"),
                                                                html.Li( str(df["Number of Families"].tail(9).sum()) +
                                                                         " Families" ),       
                                                           ])
                                ],
                                style={'display': 'inline-block', 'padding-left': '20px', 'padding-right': '30px', 'font-size' : '20px'},
                                className="card",
                       ),
                   ],
                   
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
        Input("region-filter", "value"), # For graph 1 (3 pie charts) ## This is a list
        Input("gender-filter", "value"), # For graph 1 (3 pie charts) ## This is a list
        Input("age-filter", "value"), # For graph 1 (3 pie charts)   ## This is a list
        Input("single-region-filter", "value"), ## This is a single value
        Input("month-filter", "value"),## This is a list
    ],
)
  # The  Input("region-filter", "value") line will effectively watch the "region-filter" element for changes
  # and will take its value property if the element changes
  
def update_charts(regions, gender, age, region, month):
    ## Mask cannot deal with an array/list input, so only (single) regions can go in here
    mask = (
        (df.Region == region)
    )
    ## Filtered data after region
    Single_Region_filtered_data = df.loc[mask, :] ## Will use this in creating a breakdown of counts in age ranges
    ## Filtered data after month filter
    filtered_data = df[(df['date'].isin(month))] 
    
    
    ############## Now filter the data for graph 1. ################################
    # Below Vars will be used for filtering columns
    filtered_column = "Total Adults"
    numRegions = 9
    
    ## Regions filter
    # If the region has not been selected all regions should be shown
    if(len(regions) == 0 ):
        regions=["Dublin", "Mid-East", "Midlands", "Mid-West", "North-East", 
                                       "North-West", "South-East", "South-West", "West"     
        ]
    filtered_data_1 = df[(df['Region'].isin(regions))] ## Only the regions selected are in the filtered df
    
    ## Gender Filter
    # If the user has not selected any genders the total adults figure should be used. 
    if(len(gender) == 0):
        gender = ['Male', 'Female']
    # Need to use if statements to filter Gender (as diff columns for Male and Female)
    # If the below if statement is False then no chanegs need to be made
    if (len(gender) != 2):
        filtered_column = str(gender[0]) + " Adults"
    
    ## Age bracket filter. 
    # If no age has been selected all age groups should be selected
    if(len(age) == 0 ):
        age = ["Adults Aged 18-24","Adults Aged 25-44", "Adults Aged 45-64", "Adults Aged 65+"]
    
    # if age bracket is not the default the app will create another row of graphs analyzing all the age brackets selected
    num_brackets = len(age) ## Gets the number of age brackets selected 
    if (num_brackets == 4): 
        fig = make_subplots(rows=1, cols=3, subplot_titles=[("January"),("June"),("December")] ,specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])
    else:
        fig = make_subplots(rows=(1+num_brackets), cols=3, subplot_titles=[("January"),("June"),("December")] ,specs=([[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])*(1+num_brackets) )
    
    # The first 3 graphs will only be affected by the gender and regions filters
    fig.add_trace(go.Pie( 
            values= filtered_data_1[filtered_column].head(numRegions), 
            labels= filtered_data_1["Region"].head(numRegions),
            title = filtered_column 
        ),  
        row=1, col=1
    )
    fig.add_trace(go.Pie( 
            values= filtered_data_1[filtered_column].head(numRegions*2).tail(numRegions), 
            labels= filtered_data_1["Region"].head(numRegions*2).tail(numRegions),
            title = filtered_column
        ),
        row=1, col=2
    )
    fig.add_trace(go.Pie( 
            values= filtered_data_1[filtered_column].tail(numRegions), 
            labels= filtered_data_1["Region"].tail(numRegions),
            title = filtered_column
        ),
        row=1, col=3
    )
    
    # If age brackets have been selected graphs will be generated for each bracket
    count = 2
    if(num_brackets != 4):
        for age_bracket in age:
             fig.add_trace(go.Pie(  
                    values= filtered_data_1[age_bracket].head(numRegions), 
                    labels= filtered_data_1["Region"].head(numRegions),
                    title = str(age_bracket)
                ),  
                row=count, col=1
             )
             fig.add_trace(go.Pie( 
                    values= filtered_data_1[age_bracket].head(numRegions*2).tail(numRegions), 
                    labels= filtered_data_1["Region"].head(numRegions*2).tail(numRegions),
                    title = str(age_bracket)
                ),
                row=count, col=2
             )
             fig.add_trace(go.Pie( 
                    values= filtered_data_1[age_bracket].tail(numRegions), 
                    labels= filtered_data_1["Region"].tail(numRegions),
                    title = str(age_bracket)
                ),
                row=count, col=3
             )
             count = count + 1
        
    # Will now update the length/height of the figure depending on the number of rows
    fig.update_layout(
        height = 270*count, ## If the height is any larger than 270 there is problems with 
        # how the percents appear in the top row 
    )
    
    ## Now will add a hole to make the pie charts like a donut
    fig.update_traces(hole=.4, hoverinfo="label+percent+value")
    
    ## Now will add title
    fig.update_layout(
        title_text="Breakdown of Homeless Adults by Region 2020",
    )
    
    # region_chart_figure = px.pie(df.head(9), values="Total Adults", names="Region", title= "Regional Breakdown of homeless Adults")
    region_chart_figure = fig
    ######################################## End of fig 1 #################################
    
    ####################################### Start of Fig 2 ##################################
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
