#!/usr/bin/env python
# coding: utf-8

# # IV. Create an Interactive Geographic
# Ref:
# Source:
# Source: The code is based on the work of Jim King available here: https://github.com/JimKing100/SF_Real_Estate_Live
# Info: Here we modified its code to fit our dataset and to add a google map tile

# The interactive chart below provides details on Toulouse Apartment sales.
# The chart breaks down the apartments for rent by
# Median Rental Price in €
# Average Rental Price in €
# Median Area in Square Meters
# Average Area in Square Meters
# Median Rental Price per Square Meter
# Average Rental Price per Square Meter
# Number of Apartments for Rent


# Import libraries
import json
import os

import geopandas
import numpy as np
import pandas as pd
from bokeh.io.doc import curdoc
from bokeh.layouts import widgetbox, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter
from bokeh.models import HoverTool, Select, GMapOptions
from bokeh.palettes import brewer
from bokeh.plotting import gmap

# Import google API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# ## 4.1 Load and Clean the Data

# Import the data
#neighborhood_data = pd.read_csv(r'C:/Users/jerem/Google Drive/Mes Documents/Travail/Projects/Toulouse_Apt_Rental_Price/EDA/data_seloger_EDAforSpatial_part3.csv')
neighborhood_data = pd.read_csv('https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/data/data_geomap.csv')

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
# Toulouse, through their website https://data.toulouse-metropole.fr/, has some exportable neighborhood maps in GeoJSON format providing various demographic.
# We will import one of them into a GeoDataframe object.

# Read the geojson map file for Realtor Neighborhoods into a GeoDataframe object
#tlse = geopandas.read_file(r'C:/Users/jerem/Google Drive/Mes Documents/Travail/Projects/Toulouse_Apt_Rental_Price/geomap/recensement-population-2015-grands-quartiers-population.geojson')
tlse = geopandas.read_file('https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/geomap/recensement-population-2015-grands-quartiers-population.geojson')

# First let's take a look at the neighborhoods (column 'libelle_des_grands_quartiers') displayed in nbhd_data Dataframe:
print('Numbers of unique neighborhoods in nbhd_data: {} '.format(nbhd_data['nbhd_name'].describe()))
# There are 20 unique neighborhoods in nbhd_data Dataframe.

# Now let's take a look at the neighborhoods (column 'libelle_des_grands_quartiers') displayed in tlse GeoDataframe:
print('Numbers of unique neighborhoods in tlse: {} '.format(tlse['libelle_des_grands_quartiers'].describe()))
# There are 60 unique neighborhoods in tlse GeoDataframe and thus three times more neighborhoods in the file imported from the Toulouse website.
# By taking a visual look at the neighborhood names we identify that each neighborhood have been divided in smaller ones in the GeoDataFrame.
# To fix this issue we need to:
# 1. Create a dictionary to change the neighborhood codes in the map to match the neighborhood codes in the data
# 2. Dissolve the polygons Based On an the new neighborhood codes

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

# select the columns that you wish to retain in the data
tlse_short = tlse[['nbhd_no', 'geometry']]

# then summarize the quantative columns by 'sum'
tlse_agg = tlse_short.dissolve(by='nbhd_no', aggfunc='sum')

# Convert index of a pandas dataframe into a column
tlse_agg.reset_index('nbhd_no', inplace=True)


# We use geopandas to read the geojson map into the GeoDataFrame sf.
# We then set the coordinate reference system to lat-long projection.
# Next, we rename several columns and use set_geometry to set the GeoDataFrame to column ‘geometry’ containing the active geometry (the description of the shapes to draw).
# Finally, we clean up some neighborhood id’s to match neighborhood_data.

# Set the Coordinate Referance System (crs) for projections
# ESPG code 4326 is also referred to as WGS84 lat-long projection
tlse_agg.crs = {'init': 'epsg:4326'}


# ## Create the Interactive Plot

# #### Create the JSON Data for the GeoJSONDataSource

# We now need to merges our neighborhood data with our mapping data and converts it into JSON format for the Bokeh server.

# Merge the GeoDataframe object (tlse_agg) with the neighborhood summary data (neighborhood)
merged = pd.merge(tlse_agg, nbhd_data, on='nbhd_no', how='left')

# Bokeh uses geojson formatting, representing geographical features, with json
# Convert to json
merged_json = json.loads(merged.to_json())

# Convert to json preferred string-like object
json_data = json.dumps(merged_json)


# #### Create The ColorBar

# This dictionary contains the formatting for the data in the plots
format_data = [('Tot_Apt_ForRent', 0, 500, '0,0', 'Number of Apartments For Rent'),
               ('Min_Rent', 250, 550, '0,0 ', 'Minimum Rental Price (€)'),
               ('Max_Rent', 850, 3000, '0,0', 'Maximum Rental Price (€)'),
               ('Avg_Rent', 550, 800, '0,0', 'Average Rental Price (€)'),
               ('Median_Rent', 550, 750, '0,0', 'Median Rental Price (€)'),
               ('Avg_Area', 40, 60, '0,0', 'Average Area in Square Meters'),
               ('Median_Area', 40, 60, '0,0', 'Median Area in Square Meters'),
               ('Avg_Rent_SqM', 11, 18, '0,0', 'Average Rental Price per Square Meter'),
               ('Median_Rent_SqM', 11, 18, '0,0', 'Median Rental Price per Square Meter')]

#Create a DataFrame object from the dictionary
format_df = pd.DataFrame(format_data, columns = ['field', 'min_range', 'max_range' , 'format', 'verbage'])


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
# 1. We pass it the field_name to indicate which column of data we want to plot (e.g. Median Rental Price).
# 2. Using the format_df we pull out the minimum range, maximum range and formatting for the ColorBar.
# 3. We call Bokeh’s LinearColorMapper to set the palette and range of the colorbar.
# 4. We create the ColorBar using Bokeh’s NumeralTickFormatter and ColorBar.
# 5. We create the plot figure with appropriate title.
# 6. We create the “patches”, in our case the neighborhood polygons, using Bokeh’s p.patches glyph using the data in geosource.
# 7. We add the colorbar and the HoverTool to the plot and return the plot p.


# Create a plotting function
def make_plot(field_name):
    # Set the format of the colorbar
    min_range = format_df.loc[format_df['field'] == field_name, 'min_range'].iloc[0]
    max_range = format_df.loc[format_df['field'] == field_name, 'max_range'].iloc[0]
    field_format = format_df.loc[format_df['field'] == field_name, 'format'].iloc[0]

    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    color_mapper = LinearColorMapper(palette=palette, low=min_range, high=max_range)

    # Create color bar.
    format_tick = NumeralTickFormatter(format=field_format)
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=6, formatter=format_tick,
    border_line_color=None, location=(0, 0))

    # Create figure object.
    map_options = GMapOptions(lat=43.60, lng=1.44, map_type="roadmap", zoom=12)
    verbage = format_df.loc[format_df['field'] == field_name, 'verbage'].iloc[0]
    p = gmap(GOOGLE_API_KEY, map_options,
             title=verbage + ' by Neighborhood for Apartments for Rent in Toulouse (2020)',
             plot_height=650, plot_width=850,
             toolbar_location="below")
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False

    # Add patch renderer to figure.
    p.patches('xs', 'ys', source=geosource, fill_color={'field': field_name, 'transform': color_mapper},
              line_color='black', line_width=0.25, fill_alpha=0.7)

    # Specify color bar layout.
    p.add_layout(color_bar, 'right')

    # Add the hover tool to the graph
    p.add_tools(hover)

    return p


# #### Main Code for Interactive Map
# We still need several pieces of code to make the interactive map including a ColorBar, Bokeh widgets and tools, a plotting function and an update function, but before we go into detail on those pieces let’s take a look at the main code.


# Input geojson source that contains features for plotting for:
# initial year 2018 and initial criteria sale_price_median
geosource = GeoJSONDataSource(geojson = json_data)

# Define a sequential multi-hue color palette.
palette = brewer['Reds'][9]

# Reverse color order so that dark blue is highest obesity.
palette = palette[::-1]


# #### The HoverTool
# The HoverTool is a fairly straightforward Bokeh tool that allows the user to hover over an item and display values.
# In the main code we insert HoverTool code and tell it to use the data based on the neighborhood_name and display the six criteria
# using “@” to indicate the column values.

# Add hover tool
hover = HoverTool(tooltips = [ ('Sector', '@sector_name'),
                               ('Neighborhood', '@nbhd_name'),
                               ('# Apartment', '@Tot_Apt_ForRent available'),
                               ('Median Rental Price', '@Median_Rent{,} €'),
                               ('Average Rental Price', '@Avg_Rent{,} €'),
                               ('Median Area', '@Median_Area{,} SqM'),
                               ('Median Rental Price/SqM', '@Median_Rent_SqM{0.2f} €/SqM'),
                               ('Minimum Rental Price', '@Min_Rent{,} €'),
                               ('Maximum Rental Price', '@Max_Rent{,} €')])

# Call the plotting function
input_field = 'Median_Rent'
p = make_plot(input_field)



# #### Widgets and The Callback Function
# We need to use a Bokeh widgets, more precisely a Select object allows the user to select the criteria (or column).
# This widget works on the following principle - the callback.
# In the code below, the widgets pass a ‘value’ and call a function named update_plot
# when the .on_change method is called (when a change is made using the widget - the event handler).

# Make a selection object: select
select = Select(title='Select Criteria:', value='Median Rental Price (€)',
              options=['Median Rental Price (€)', 'Average Rental Price (€)',
                       'Median Rental Price per Square Meter', 'Average Rental Price per Square Meter',
                       'Median Area in Square Meters', 'Average Area in Square Meters',
                       'Number of Apartments For Rent'])

select.on_change('value', update_plot)


# Make a column layout of widgetbox(slider) and plot, and add it to the current document
# Display the current document
layout = column(p, widgetbox(select))
curdoc().add_root(layout)

# #### The Static Map with ColorBar and HoverTool
# Show the map
#show(p)

# Save the map
#outfp = r'./geomap/test3.html' # Output filepath
#save(p, outfp)


# #### The Bokeh Server
#from bokeh import server
# Run from command prompt
# cd C:\Users\jerem\Google Drive\Mes Documents\Travail\Projects\Toulouse_Apt_Rental_Price\geomap
# bokeh serve --show Apartment_Rental_Price_Prediction_202001_GeoMap_Part4.py
