# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:36:19 2021

@author: micha
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:38:43 2021

@author: micha
"""
################# Michael & Jasons Dashboard ###########################
######### This dashboard examines the homelessness situation in Ireland in 2020 #############

## When you run the program search http://127.0.0.1:8050/ and the 
#  interactive dashboard should be visible and working 

## Good source: https://realpython.com/python-dash/

import dash #  helps you initialize your application (dashboard) 
import dash_core_components as dcc # allows you to create interactive components like graphs, dropdowns, or date ranges
import dash_html_components as html # lets you access HTML tags
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# Originally planned to do a regional heatmap, but it was too hard
#import branca.colormap as cm
#import folium
#import geopandas as gpd 

###################################### Read in the datasets from Data.Gov.ie #############################
from urllib.request import urlretrieve
import sys ## For sys.exit()

### Read in as csv file directly from the data.Gov.ie site
######### Jan 2020 Report
url_1 = "https://data.usmart.io/org/ae1d5c14-c392-4c3f-9705-537427eeb413/resource?resourceGUID=4ef0ffc4-2ff8-429f-aed4-24c5f2189d62"
## Need to have an exception hander in case the data won't load
try:
    urlretrieve (url_1, "Homelessness Report January 2020.csv")
except: 
    # I didn't use the following commented out code as I don't think the errors outputted are very descriptive in this case
    # except Exception as exc:
    # print('There was a problem: %s' % (exc))
    print("Unable to read in the 'Homelessness Report January 2020.csv' dataset.\n" + 
          "Please check your internet connection. Alternatively the path to the file on"+
          " data.gov.ie may have changed or the site may be down.")
    print("\nProgram Execution aborted. Goodbye.")
    sys.exit() ## This will terminate the script execution
    ### Note: if you run code chunk by chunk and it arrives at a sys.exit an error will be thrown.
    ## If the whole script is ran it will work as intended.
datasetJan20 = pd.read_csv(url_1) ## Read in as pandas datafile

######### June 2020 report
url_2 = "https://data.usmart.io/org/ae1d5c14-c392-4c3f-9705-537427eeb413/resource?resourceGUID=a1880631-b402-4bc1-a340-c88f8e53912b"
try:
    urlretrieve (url_2, "Homelessness Report June 2020.csv")   
except:
    print("Unable to read in the 'Homelessness Report June 2020.csv' dataset.\n" + 
          "Please check your internet connection. Alternatively the path to the file on"+
          " data.gov.ie may have changed or the site may be down.")
    print("\nProgram Execution aborted. Goodbye.")
    sys.exit() ## This will terminate the script execution
datasetJune20 = pd.read_csv(url_2)

####### Dec 2020 report
url_3 = "https://data.usmart.io/org/ae1d5c14-c392-4c3f-9705-537427eeb413/resource?resourceGUID=2827f749-43bc-46d9-9f55-4e3aa035aa5d"
try:
    urlretrieve (url_3, "Homelessness Report December 2020.csv")   
except:
    print("Unable to read in the 'Homelessness Report December 2020.csv' dataset.\n" + 
          "Please check your internet connection. Alternatively the path to the file on"+
          " data.gov.ie may have changed or the site may be down.") 
    print("\nProgram Execution aborted. Goodbye.")
    sys.exit() ## This will terminate the script execution
datasetDec20 = pd.read_csv(url_3)


############## (Data read in here NOT USED) Reading the data from data.gov.ie API ################
### The below code successfully reads in the January 2020 dataset from the API.

# I did not use this code for reading in any of the 3 datasets that were used in the (dashboard) app as
# I belive its more complicated that the previous method of reading in the datasets. 
# An increased level of complexity normally results in there being more errors. 

## I left the code in just to show that we were able to interrogate and work with the API.
# It prob would be sensible to take it out considering errors may be thrown from this and the program
# will be terminated when it may have been able to proceed (i.e. data was read in correctly above),
# but I just wanted to show we could do it. 
import json
import requests

url_API = "https://api.usmart.io/org/ae1d5c14-c392-4c3f-9705-537427eeb413/29df500e-877c-4146-9f9e-0182004bac7c/1/urql"
res = requests.get(url_API)

### Now check if request was successful 
try:
    res.raise_for_status()
except Exception as exc:
    print('There was a problem: %s' % (exc))
    sys.exit() ## This will terminate the script execution
    
JSONContent = res.json()
content = json.dumps(JSONContent, indent = 4, sort_keys = True)
### Now create a Pandas df from the JSON
jsonData = json.loads(content) ## Change the (JSON) string into a JSON object that can be read into a pd df
test_df = pd.DataFrame.from_dict(jsonData)
## Drop an uneccessary column that appears to be an API id tag of some sorts
test_df = test_df.drop(['usmart_id'], axis=1)
## This df is unused in the (dashboard) app

##########################################################################
######################### End of Reading in the Data ########################
##########################################################################

######################## Clean & Merge the data ##########################
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


##############################################################################################################
############################## Now create a Plotly Dash Dashboard ####################################
##############################################################################################################
################### Styling
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
# the head tag of the application.
# Alternatively the stylying could have been done like so:
    #html.H1(
     #    children = "Homelessness 2020",
      #   style = {"fontSize":"48px", "color":"blue"},   
    #),
## Doing it the above way is very slow and tedious however, so I used a CSS sheet


########## Create app ####################
app = dash.Dash(__name__, external_stylesheets = external_stylesheets) ## creates instance of the dash class
# server = app.server # Tried releasing the app on Heroku but can't get it to work right on that platform
app.title = "Homelessness Dashboard" ## This is the text that will appear in the title bar of your web browser 

############ Define the layout of the dash application ####################
app.layout = html.Div(
    children = [
        html.Div(
            ############### Page (Dashboard) Title #########################
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
        ############################ First set of Filters: ##########################
        # (When the user uses these filters the graphs that can be filtered by this data will be changed,
        #  code for this is in an update function near the end of the script.)
        # Note: Had to be very careful with filters as for example it is not possible to know from the dataset how many 
        # females from Dublin accessed private emergency accomodation
        html.Div(
            children=[
                html.Div( 
                    ########### Regions of Ireland Filter
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
                            multi=True, ### Can select multiple
                            className="dropdown",
                        ),
                    ]
                ),
                 html.Div(
                    ############ Gender Filter
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
                    ############## Age selector
                    # (When this selector is used more graphs will appear below the 3 donut pie charts)
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
        ##################### End of first set of filters #################
        html.Div(
            children=[
                ####################### Fig 1. The region donut pie-charts. ###################
                html.Div(
                    children=dcc.Graph(
                        id="Region-chart", config={"displayModeBar" : False},
                        ######## Note the fig element is created in the update function so that
                        # it is interactive
                    ),
                    className="card"
                ),
                ####################### End of fig 1 ########################
                
                ########################## Second set of filters ##################
                html.Div(
                    children=[
                        html.Div(
                            ############ Single Region of Ireland Filter
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
                            ########## Date selector filter. 
                            # This was orginally a calander filter, but as there is only
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
                                    value="January", ## Default (initial) Value. 
                                    clearable=False,
                                    multi=False, ## Prevent user from selecting multiple as graphing becomes very challenging. 
                                    className="dropdown",
                                ),
                            ]
                        ),
                    ],
                    className="second-menu",
                ),
                ###################### End of 2nd set of filters ########################
                
                ##################### Remaining graphs (Fig 2,3,4) #####################
                html.Div(
                    children=dcc.Graph(
                        id="Adult-chart", config={"displayModeBar" : False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="Accomodation-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="National-families-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                ##################### End of remaining graphs #################
                
                #################### National Summary statistics #################
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

#### Making the Dashboard interative
## Using a callback to make the grapgs interactive. The inputs and outputs are clearly defined here. 
## These onjects have two arguments:
    # The indentifier element that they'll modify when the function executes
    # The property of the element to be modified. 
@app.callback(
    [Output("Region-chart", "figure"), Output("Adult-chart", "figure"), Output("Accomodation-chart", "figure"), Output("National-families-chart", "figure")],
    [
        Input("region-filter", "value"), # For graph 1 (3 pie charts) ## This is a list
        Input("gender-filter", "value"), # For graph 1 (3 pie charts) ## This is a list
        Input("age-filter", "value"), # For graph 1 (3 pie charts)   ## This is a list
        Input("single-region-filter", "value"), ## For graph 2  ## This is a single value
        Input("month-filter", "value"),## For graph 2   ## This is a single value
    ],
)
  # The  Input("region-filter", "value") line will effectively watch the "region-filter" element for changes
  # and will take its value property if the element changes
  
  
### The update function is used to make the graphs interactive
def update_charts(regions, gender, age, region, month):
       
    ################# Now filter the data for graph 1. ################################
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
    ########################### End of filtering data for fig 1 ###############
    
    ########################### Create fig 1 ##################################
    ### Will create a number of donut pie-charts using make_suplots function.
    
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
    
    
    
    ####################################### Filter data for fig 2 #########################
    ## Mask cannot deal with an array/list input, so only (single) regions can go in here
    mask = (
        (df.Region == region)
        & (df.date == month)
    )
    
    ## Filtered data after region and month filter applied 
    filtered_data_2 = df.loc[mask, :]
   
    ####################################### End of filtering data for fig 2 ################
    
    ####################################### Start of Fig 2 ##################################
    
    pv = pd.pivot_table(filtered_data_2, index=['Region'], values=['Adults Aged 18-24', 'Adults Aged 25-44', 'Adults Aged 45-64', 'Adults Aged 65+'], aggfunc=sum, fill_value=0) 
   
    fig2 = make_subplots(rows=1, cols=1)
    fig2.add_trace(
        go.Bar(
            x=pv.columns, 
            y=pv.values[0], 
            name = "Adults Aged 18-24",
            ## Now set colors of the columns
            marker_color='rgb(158,202,225)', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    ## Now will add title
    fig2.update_layout( 
        title_text="Breakdown of Homeless Adults in " + str(pv.index.values[0]) + ", " + str(month) + " 2020" + " by age bracket",
        #xaxis =  {'showgrid': False},
        #yaxis = {'showgrid': True, 'color':'black'},
        plot_bgcolor='rgba(0,0,0,0)' ## Set the background colour white  
    )
     
    adult_chart_figure = fig2
    #################################### End of fig 2 ####################

    ################################### Start of fig 3 #################
    ## Fig 3 uses the same filtered data as fig 2
    pv_2 = pd.pivot_table(filtered_data_2, index=['Region'], 
                          values=['Number of people who accessed Private Emergency Accommodation',
                                    'Number of people who accessed Supported Temporary Accommodation', 
                                    'Number of people who accessed Temporary Emergency Accommodation', 
                                    'Number of people who accessed Other Accommodation'], 
                          aggfunc=sum, fill_value=0) 
   
    fig3 = make_subplots(rows=1, cols=1)
    fig3.add_trace(
        go.Bar(
            x=["Private Emergency Acc.","Supported Temp Acc.", "Temp Emergency Acc.", "Other Acc."], 
            y=pv_2.values[0],  
            ## Now set colors of the columns
            marker_color='PaleGreen',  
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    ## Now will add title
    fig3.update_layout( 
        title_text="Number of times different types of Accomodation were accessed in " + str(pv_2.index.values[0] + ", "+ str(month) + " 2020"),
        #xaxis =  {'showgrid': False},
        #yaxis = {'showgrid': True, 'color':'black'},
        plot_bgcolor='rgba(0,0,0,0)', ## Set the background colour white  
        height = 500,
        width = 1000
    )
    
    accomodation_chart_figure = fig3
    ############################## End of fig 3 #############################
    
    ############################# Start of fig 4 ###########################
    ##### Filter the data only with respect to month as want to create a national graph:
    mask_2 = (
        (df.date == month)
    )
    ## Filtered data after month filter applied 
    filtered_data_3 = df.loc[mask_2, :]
    
    ##### Now create fig:
    pv_3 = pd.pivot_table(filtered_data_3, index=['Region'], values=['Number of Families', 
                                                                     'Number of Adults in Families', 
                                                                     'Number of Single-Parent families', 
                                                                     'Number of Dependants in Families'
                                                                     ], aggfunc=sum, fill_value=0) 
   
    fig4 = make_subplots(rows=1, cols=1)
    ### I will now create a trace for each Region
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[0], 
            name = pv_3.index[0],
            ## Now set colors of the columns
            marker_color= 'green', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[1], 
            name = pv_3.index[1],
            ## Now set colors of the columns
            marker_color='Chartreuse', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[2], 
            name = pv_3.index[2],
            ## Now set colors of the columns
            marker_color='cyan', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[3], 
            name = pv_3.index[3],
            ## Now set colors of the columns
            marker_color='Coral', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[4], 
            name = pv_3.index[4],
            ## Now set colors of the columns
            marker_color='Crimson', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[5], 
            name = pv_3.index[5],
            ## Now set colors of the columns
            marker_color='Chocolate', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[6], 
            name = pv_3.index[6],
            ## Now set colors of the columns
            marker_color='DarkSlateBlue', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[7], 
            name = pv_3.index[7],
            ## Now set colors of the columns
            marker_color='Dark Orange', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    fig4.add_trace(
        go.Bar(
            x=pv_3.columns, 
            y=pv_3.values[8], 
            name = pv_3.index[8],
            ## Now set colors of the columns
            marker_color='Violet', 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.6
        )
    )
    ## Now will add title
    fig4.update_layout( 
        title_text="National Homeless Family figures in " + str(month) + " 2020",
        #xaxis =  {'showgrid': False},
        #yaxis = {'showgrid': True, 'color':'black'},
        plot_bgcolor='rgba(0,0,0,0)' ## Set the background colour white  
    )
    national_family_chart_figure = fig4
    
    ### Now return the graphs 
    return region_chart_figure, adult_chart_figure, accomodation_chart_figure, national_family_chart_figure 

 


#################### Run application ####################### 
## The following two lines of code run the application locally using Flask’s built-in server.
if __name__ == "__main__":
    app.run_server(debug=False) #  (***NOW DISABLED (As debug = False)****) If True enables the hot-reloading option in the application. 
                               #  This means that when changes are made to the app,
                               #  it reloads automatically, without having to restart the server.
## When this is run a web page should load. 

############################################### Should Prob disable the hot-reloading when submit *******************************************