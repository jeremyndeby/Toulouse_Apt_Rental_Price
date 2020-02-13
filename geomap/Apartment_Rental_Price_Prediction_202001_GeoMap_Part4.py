#!/usr/bin/env python
# coding: utf-8

# # IV. Create an Interactive Geographic
# Ref: 
# - https://towardsdatascience.com/how-to-create-an-interactive-geographic-map-using-python-and-bokeh-12981ca0b567

# If you are looking for a powerful way to visualize geographic data then you should learn to use interactive Choropleth maps. A Choropleth map represents statistical data through various shading patterns or symbols on predetermined geographic areas such as countries, states or counties. Static Choropleth maps are useful for showing one view of data, but an interactive Choropleth map is much more powerful and allows the user to select the data they prefer to view.
# 
# The interactive chart below provides details on San Francisco single family homes sales. The chart breaks down the single family home sales by Median Sales Price, Minimum Income Required, Average Sales Price, Average Sales Price Per Square Foot, Average Square Footage and Number of Sales all by neighborhood and year (10 years of data).

# In[1]:


import os
import pandas as pd
from IPython.display import display
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
from matplotlib.colors import LogNorm
from scipy.stats import skew

# import necessary packages to work with spatial data in Python



pd.options.display.max_columns = None
pd.options.display.max_rows = None


# ## Using Python and Bokeh
# After exploring several different approaches, I found the combination of Python and Bokeh to be the most straightforward and well-documented method for creating interactive maps.
# 
# Let’s start with the installs and imports you will need for the graphs. Pandas, numpy and math are standard Python libraries used to clean and wrangle the data. The geopandas, json and bokeh imports are libraries needed for the mapping.
# 
# I work in Colab and needed to install fiona and geopandas.

# In[2]:


# Import libraries
import pandas as pd
import numpy as np
import math

import fiona
import geopandas
import json

from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter
from bokeh.palettes import brewer

from bokeh.io.doc import curdoc
from bokeh.models import Slider, HoverTool, Select
from bokeh.layouts import widgetbox, row, column


# 

# ## 4.1 Load and Clean the Data

# Here we are importing the data from the csv file

# In[3]:


neighborhood_data = pd.read_csv('data_seloger_EDAforSpatial_part3.csv')
neighborhood_data.head()


# After loading the dataset creating in the previous part we need to clean our data as we want to be able to see on the map for each neighborhood:
# - The total number of appartments listed
# - The lowest rent
# - The highest rent
# - The average rent
# - The median rent
# - The average area in square meters
# - The median area in square meters
# - The average rent per square meters
# - The median rent per square meters
# 
# 
# A rent per square meters feature is added to neighborhood_data and the dataframe is summarized using groupby and aggregate functions to create the final nbhd_data dataframe with all numeric fields converted to integer values for ease in displaying the data (except for Avg_Rent_SqM and Median_Rent_SqM, we will round them to the first decimal):

# In[4]:


# Create a rent_SqM feature
neighborhood_data['Rent_SqM'] = neighborhood_data['rent'] / neighborhood_data['area']

# Create a new dataframe with the new features of interest
#Max_Rent = round(neighborhood_data.groupby('nbhd_no').rent.max(),0)
nbhd_data = neighborhood_data.groupby(['sector_no', 'sector_name', 'nbhd_no', 'nbhd_name']).agg(Tot_Apt_ForRent=('nbhd_no', 'size'),
                                              Min_Rent=('rent', 'min'), 
                                              Max_Rent=('rent', 'max'), 
                                              Avg_Rent=('rent', np.mean), 
                                              Median_Rent=('rent', np.median), 
                                              Avg_Area=('area', np.mean), 
                                              Median_Area=('area', np.median), 
                                              Avg_Rent_SqM=('Rent_SqM', np.mean), 
                                              Median_Rent_SqM=('Rent_SqM', np.median))

# Convert index of a pandas dataframe into a column
nbhd_data.reset_index('nbhd_name', inplace=True)
nbhd_data.reset_index('nbhd_no', inplace=True)
nbhd_data.reset_index('sector_name', inplace=True)
nbhd_data.reset_index('sector_no', inplace=True)

# Convert to integer
cols_round0 = ['Tot_Apt_ForRent', 
               'Min_Rent', 'Max_Rent', 'Avg_Rent','Median_Rent', 
               'Avg_Area', 'Median_Area']
for i in cols_round0:
    nbhd_data = nbhd_data.astype({i: 'int'})
    
# Round to the first decimal 
cols_round1 = ['Avg_Rent_SqM','Median_Rent_SqM']
nbhd_data[cols_round1] = nbhd_data[cols_round1].round(1)

nbhd_data.sort_values(by=['nbhd_no'])


# We now need to map this data onto a Toulouse neighborhood map.

# ## Prepare the Mapping Data and GeoDataFrame
# "We will be working with GeoJSON, a popular open standard for representing geographical features with JSON. JSON (JavaScript Object Notation), is a minimal, readable format for structuring data. Bokeh uses JSON to transmit data between a bokeh server and a web application.
# 
# In a typical Bokeh interactive graph the data source needs to be a ColumnDataSource. This is a key concept in Bokeh. However, when using a map we use a GeoJSONDataSource instead.
# 
# To make our work with geospatial data in Python easier we use GeoPandas. It combines the capabilities of pandas and shapely, providing geospatial operations in pandas and a high-level interface to multiple geometries to shapely. We will use GeoPandas to create a GeoDataFrame - a precursor to creating the GeoJSONDataSource." Jim King

# Finally, we need a map that is in GeoJSON format. Toulouse, through their website https://data.toulouse-metropole.fr/, has some exportable neighborhood maps in GeoJSON format providing various demographic. We will import one of them into a GeoDataframe object.

# In[5]:


# Read the geojson map file for Realtor Neighborhoods into a GeoDataframe object
tlse = geopandas.read_file('recensement-population-2015-grands-quartiers-population.geojson')
tlse.head()


# A key column of the data is the neighborhood code (grd_quart) which needs to match the mapping code for the neighborhood. This will allow us to merge the data with the map. Before merging the data we then need to make sure the neighborhoods in both files do match.

# First let's take a look at the neighborhoods (column 'libelle_des_grands_quartiers') displayed in nbhd_data Dataframe:

# In[6]:


print('Numbers of unique neighborhoods in nbhd_data: {} '.format(nbhd_data['nbhd_name'].describe()))


# There are 20 unique neighborhoods in nbhd_data Dataframe. 
# Now let's take a look at the neighborhoods (column 'libelle_des_grands_quartiers') displayed in tlse GeoDataframe:

# In[7]:


print('Numbers of unique neighborhoods in tlse: {} '.format(tlse['libelle_des_grands_quartiers'].describe()))


# There are 60 unique neighborhoods in tlse GeoDataframe and thus three times more neighborhoods in the file imported from the Toulouse website. By taking a visual look at the neighborhood names we identify that each neighborhood have been divided in smaller ones in the GeoDataFrame.  To fix this issue we need to: 
# 1. Create a dictionary to change the neighborhood codes in the map to match the neighborhood codes in the data
# 2. Dissolve the polygons Based On an the new neighborhood codes

# In[59]:


# Create a dictionary to change the neighborhood codes to neighborhood names.
nbhd_dict = {'3155507': 'n1_2','3155533': 'n2_3','3155502': 'n1_1','3155532': 'n2_3',
             '3155537': 'n3_1','3155556': 'n6_2','3155552': 'n6_3','3155519': 'n4_1',                            
             '3155501': 'n1_1','3155505': 'n1_1','3155535': 'n2_4','3155545': 'n4_3',                            
             '3155508': 'n1_2','3155522': 'n4_3','3155540': 'n3_3','3155528': 'n2_2',
             '3155527': 'n5_3','3155530': 'n2_3','3155515': 'n2_1','3155531': 'n2_3',
             '3155536': 'n3_2','3155541': 'n4_2','3155521': 'n4_3','3155526': 'n5_2',                        
             '3155543': 'n4_2','3155534': 'n6_2','3155551': 'n6_4','3155546': 'n5_1',
             '3155538': 'n3_2','3155558': 'n6_2','3155512': 'n5_3','3155509': 'n1_3',
             '3155539': 'n3_3','3155557': 'n6_2','3155520': 'n4_1','3155510': 'n1_3',
             '3155518': 'n3_1','3155554': 'n6_3','3155547': 'n5_1','3155529': 'n2_3',
             '3155516': 'n2_1','3155523': 'n5_1','3155549': 'n5_3','3155560': 'n6_1',
             '3155514': 'n5_3','3155548': 'n5_2','3155553': 'n6_3','3155542': 'n4_2',
             '3155525': 'n5_3','3155511': 'n1_3','3155506': 'n2_1','3155504': 'n1_1',               
             '3155503': 'n1_1','3155559': 'n6_1','3155513': 'n5_3','3155555': 'n6_4',
             '3155524': 'n5_2','3155517': 'n3_2','3155544': 'n4_2','3155550': 'n6_4'}

# Create a neighborhood name from the dictionary neighborhood_dict
tlse['nbhd_no'] = tlse['grd_quart'].map(nbhd_dict)


# In[60]:


# select the columns that you wish to retain in the data
tlse_short = tlse[['nbhd_no', 'geometry']]

# then summarize the quantative columns by 'sum' 
tlse_agg = tlse_short.dissolve(by='nbhd_no', aggfunc = 'sum')

# Convert index of a pandas dataframe into a column
tlse_agg.reset_index('nbhd_no', inplace=True)

tlse_agg.columns


# We use geopandas to read the geojson map into the GeoDataFrame sf. We then set the coordinate reference system to lat-long projection. Next, we rename several columns and use set_geometry to set the GeoDataFrame to column ‘geometry’ containing the active geometry (the description of the shapes to draw). Finally, we clean up some neighborhood id’s to match neighborhood_data.

# In[97]:


# Set the Coordinate Referance System (crs) for projections
# ESPG code 4326 is also referred to as WGS84 lat-long projection
tlse_agg.crs = {'init': 'epsg:4326'}


# We now have our neighborhood data in nbhd_data and our mapping data in tlse with both sharing the neighborhood code column subdist_no.

# ## Create the Interactive Plot

# #### Create the JSON Data for the GeoJSONDataSource

# We now need to merges our neighborhood data with our mapping data and converts it into JSON format for the Bokeh server.

# In[62]:


# Merge the GeoDataframe object (tlse_agg) with the neighborhood summary data (neighborhood)
merged = pd.merge(tlse_agg, nbhd_data, on='nbhd_no', how='left')

# Bokeh uses geojson formatting, representing geographical features, with json
# Convert to json
merged_json = json.loads(merged.to_json())

# Convert to json preferred string-like object 
json_data = json.dumps(merged_json)


# #### Create The ColorBar
# The ColorBar is “attached” to the plot and the entire plot needs to be refreshed when a change in the criteria is requested. Each criteria has it’s own unique minimum and maximum range, format for displaying and verbage. For example, Number of Appartment For Rent has a range of 0–100, a format as an integer and the name 'Number of Appartment For Rent' that needs to be changed in the title of the plot.
# 
# So we need to create a format_df that details the data needed in the ColorBar and title.

# In[63]:


merged.describe()


# In[126]:


# This dictionary contains the formatting for the data in the plots
format_data = [('Tot_Apt_ForRent', 0, 500, '0,0', 'Number of Appartment For Rent'),
               ('Min_Rent', 250, 550, '$0,0 ', 'Minimum Rent'),
               ('Max_Rent', 850, 3000, '0,0', 'Maximum Rent'),
               ('Avg_Rent', 550, 800, '$0,0', 'Average Rent'),
               ('Median_Rent', 550, 750, '$0,0', 'Median Rent'),
               ('Avg_Area', 40, 60, '0,0', 'Average Area in Square Meters'),
               ('Median_Area', 40, 60, '0,0', 'Median Area in Square Meters'),
               ('Avg_Rent_SqM', 11, 18, '$0,0', 'Average Rent per Square Meter'),
               ('Median_Rent_SqM', 11, 18, '$0,0', 'Median Rent per Square Meter')]
 
#Create a DataFrame object from the dictionary 
format_df = pd.DataFrame(format_data, columns = ['field' , 'min_range', 'max_range' , 'format', 'verbage'])


# The callback function update_plot has three parameters. The attr parameter is simply the ‘value’ you passed (e.g. slider.value or select.value), the old and new are internal parameters used by Bokeh and you do not need to deal with them.
# 
# We select re-set the input_field (Select) based on criteria (cr) before re-seting the plot based on the current input_field.

# In[127]:


# Define the callback function: update_plot
def update_plot(attr, old, new):
    # The input cr is the criteria selected from the select box
    cr = select.value
    input_field = format_df.loc[format_df['verbage'] == cr, 'field'].iloc[0]
    
    # Update the plot based on the changed inputs
    p = make_plot(input_field)
    
    # Update the layout, clear the old document and display the new document
    layout = column(p, widgetbox(select))
    curdoc().clear()
    curdoc().add_root(layout)
    
    # Update the data
    geosource.geojson = json_data 


# #### Create a Plotting Function
# The final piece of the map is make_plot, the plotting function. Let’s break this down:
# 1. We pass it the field_name to indicate which column of data we want to plot (e.g. Median Sales Price).
# 2. Using the format_df we pull out the minimum range, maximum range and formatting for the ColorBar.
# 3. We call Bokeh’s LinearColorMapper to set the palette and range of the colorbar.
# 4. We create the ColorBar using Bokeh’s NumeralTickFormatter and ColorBar.
# 5. We create the plot figure with appropriate title.
# 6. We create the “patches”, in our case the neighborhood polygons, using Bokeh’s p.patches glyph using the data in geosource.
# 7. We add the colorbar and the HoverTool to the plot and return the plot p.

# In[128]:


# Create a plotting function
def make_plot(field_name):    
    # Set the format of the colorbar
    min_range = format_df.loc[format_df['field'] == field_name, 'min_range'].iloc[0]
    max_range = format_df.loc[format_df['field'] == field_name, 'max_range'].iloc[0]
    field_format = format_df.loc[format_df['field'] == field_name, 'format'].iloc[0]

    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    color_mapper = LinearColorMapper(palette = palette, low = min_range, high = max_range)

    # Create color bar.
    format_tick = NumeralTickFormatter(format=field_format)
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=18, formatter=format_tick,
    border_line_color=None, location = (0, 0))

    # Create figure object.
    verbage = format_df.loc[format_df['field'] == field_name, 'verbage'].iloc[0]

    p = figure(title = verbage + ' by Neighborhood for Appartments for Rent in Toulouse (2020)', 
             plot_height = 650, plot_width = 850,
             toolbar_location = None)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False

    # Add patch renderer to figure. 
    p.patches('xs','ys', source = geosource, fill_color = {'field' : field_name, 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)

    # Specify color bar layout.
    p.add_layout(color_bar, 'right')

    # Add the hover tool to the graph
    p.add_tools(hover)
    return p


# #### Main Code for Interactive Map
# We still need several pieces of code to make the interactive map including a ColorBar, Bokeh widgets and tools, a plotting function and an update function, but before we go into detail on those pieces let’s take a look at the main code.

# In[194]:


# Input geojson source that contains features for plotting for:
# initial year 2018 and initial criteria sale_price_median
geosource = GeoJSONDataSource(geojson = json_data)

# Define a sequential multi-hue color palette.
palette = brewer['Purples'][9] # "La Garonne est viola"

# Reverse color order so that dark blue is highest obesity.
palette = palette[::-1]


# #### The HoverTool
# The HoverTool is a fairly straightforward Bokeh tool that allows the user to hover over an item and display values. In the main code we insert HoverTool code and tell it to use the data based on the neighborhood_name and display the six criteria using “@” to indicate the column values.

# In[191]:


# Add hover tool
hover = HoverTool(tooltips = [ ('Sector','@sector_name'),
                               ('Neighborhood','@nbhd_name'),
                               ('#Apt. For Rent', '@Tot_Apt_ForRent available'),
                               ('Average Rent', '@Avg_Rent{,} €'),
                               ('Median Rent', '@Median_Rent{,} €'),
                               ('Median Area', '@Median_Area{,} SqM'),
                               ('Median Rent/SqM', '@Median_Rent_SqM{0.2f} €/SqM'),
                               ('Minimum Rent', '@Min_Rent{,} €'),
                               ('Maximum Rent', '@Max_Rent{,} €')])
               
# Call the plotting function
input_field = 'Median_Rent_SqM'
p = make_plot(input_field)


# #### Widgets and The Callback Function
# We need to use a Bokeh widgets, more precisely a Select object allows the user to select the criteria (or column).
# This widget works on the following principle - the callback. 
# In the code below, the widgets pass a ‘value’ and call a function named update_plot when the .on_change method is called (when a change is made using the widget - the event handler).

# In[192]:


# Make a selection object: select
select = Select(title='Select Criteria:', value='Median Sales Price', options=['Median Rent', 'Average Rent',
                                                                               'Median Rent per Square Meter', 'Average Rent per Square Meter',
                                                                               'Median Area in Square Meters', 'Average Area in Square Meters',
                                                                               'Minimum Rent','Maximum Rent',
                                                                               'Number of Appartment For Rent'])

select.on_change('value', update_plot)


# #### The Static Map with ColorBar and HoverTool

# In[196]:


# Use the following code to test in a notebook, comment out for transfer to live site
# Interactive features will not show in notebook
output_notebook()
show(p)


# In[197]:


output_file('test.html')
show(p)


# #### The Bokeh Server
# I developed the static map using 2018 data and Median Sales Price in Colab in order to get the majority of the code working prior to adding the interactive portions. In order to test and view the interactive components of Bokeh, you will need to follow these steps.
# 1. Install the Bokeh server on your computer.
# 2. Download the .ipynb file to a local directory on your computer.
# 3. From the terminal change the directory to the directory with the .ipynb file.
# 4. From the terminal run the following command: bokeh serve (two dashes)show filename.ipynb
# 5. This should open a local host on your browser and output your interactive graph. If there is an error it should be visible in the terminal.

# In[198]:





# In[170]:


bokeh serve:show filename.ipynb


# In[ ]:


import Jinja2
import packaging
import pillow
import dateutil
import PyYAML
import six
import tornado
import Futures
import bokeh


# #### Public Access to the Interactive Graph via Heroku
# Once you get the interactive graph working locally, you can let others access it by using a public Bokeh hosting service such as Heroku. Heroku will host the interactive graph allowing you to link to it (as in this article) or use an iframe such as on my GitHub Pages site.
# The basic steps to host on Heroku are:
# 1. Change the Colab notebook to comment out the install of fiona and geopandas. Heroku has these items and the build will fail if they are in the code.
# 2. Change the Colab notebook to comment out the last two lines (output_notebook() and show(p)).
# 3. Download the Colab notebook as a .py file and upload it to a GitHub repository.
# 4. Create a Heroku app and connect to your GitHub repository containing your .py file.
# 5. Create a Procfile and requirements.txt file. See mine in my GitHub.
# 6. Run the app!
# 

# In[ ]:




