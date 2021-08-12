# PythonProjects
This repo will store projects from a 3rd year Python Module from 18/03/21. *****See the video DemoOfPythonDashboard.mp4 in this repo for demo of created Python dashboard.***** 

Final project:
  - script 1 (location: finalProject/Script1/Script1ReadFile.py): 
  - 
  Allows user to use and interact with datasets on data.gov.ie. Read in files that the user has downloaded by allowing the user 
  to select them with a GUI (graphical user interface). Program accepts the following file formats: csv, json, json-stat. 
  The program extracts the data and save the data in a suitable format, provides some descriptive statistics about the dataset and exports the data to an excel spreadsheet. 
  
  - script 2 (location: app.py)**: 
  *****See DemoOfPythonDashboard.mp4 in this repo.***** 
  
  Created a dashboard from 3 datasets on data.gov.ie. Datasets were read in from URLs using urrlib, code was also written to read the datasets in from the data.gov.ie API. 
  The datasets were stored as Pandas dataframes and the data was then cleaned and merged.
  I then created a Plotly Dash Dashboard, and used an external CSS stylesheet to style the created webpage. 
  Several interactive graphs were then created to visualize the data, the homeless situation in Ireland during 2020.
  When this program is run enter 'http://127.0.0.1:8050/' in your web browser and the dashboard should be visible and interactive. 
