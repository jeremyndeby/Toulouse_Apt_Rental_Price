# Toulouse_Apt_Rental_Price:

Under Construction

## Introduction: About the Project

Thousands of new students move to Toulouse (France) every year for their studies for a total. It is often difficult for them to know which neighborhoods are the cheapest or to identify which apartments have been undervalued/overvalued based on their set of features. However, many research and tutorials have been focusing on training models to predict property selling prices, but relatively few on predicting apartment rental prices. This project aims to fill the gap and provide useful insights to people, especially students, to help them in their decision-making as to which apartment to rent out.

In summary, the aims of this project are: 
1. To build a web scraper to collect the apartment listings in Toulouse from the platform SeLoger.com.
2. To build a model to estimate what should be the correct rental price given different features and their property.
3. To provide an interactive geographic map to get more familiar with the different neighborhoods of the city. 


## Interactive Map
The interactive chart below provides details on Toulouse apartments for rent currently available. The chart breaks down the apartments for rent by Median Sales Price, Minimum Income Required, Average Sales Price, Average Sales Price Per Square Foot, Average Square Footage and Number of Sales all by neighborhood.
[Interactive Chart of Toulouse Apartments For Rent 2020](https://toulouse-apt-rental-price.herokuapp.com/geomap)


## Data: A data directory
In Toulouse, SeLoger.com is an online marketplace allowing real estate agencies and owners to post listings on their website. The website gathers most of apartments for rent of the city. The data is based on listings from SeLoger.com and is collected using a custom scraper. 
- tttt.csv - The raw data scraped from https://www.seloger.com/immobilier/locations/immo-toulouse-31/bien-appartement/?LISTING-LISTpg=0 for all currently available apartments for rent in Toulouse, France.
- data_seloger_EDA_part3.csv - The cleaned data after processing and cleaning steps.
- data_seloger_EDAforSpatial_part3.csv - The cleaned dataset to use for the mapping part.
- recensement-population-2015-grands-quartiers-population.geojson - The mapping data downloaded from https://data.toulouse-metropole.fr/explore/dataset/recensement-population-2015-grands-quartiers-population/export/

## Notebooks

There are three notebooks within this repository representing different stages of the project:
1. scraper.py - Scrape the data from SeLoger.com
2. data_processing.py - Prepare raw data scraped from 
3. EDA.py - EDA of the cleaned data
4. model.py - Build and compare several models and find the best hyper-parameters for the final model
5. geomap.py - Create an interactive Geographical Map from the cleaned dataset

Also included is a pdf of a brief presentation summarizing the study and its results.
- Toulouse_Map_Code_Final.py - The colab mapping code for the interactive Toulouse Real Estate Map.
- Toulouse_Map_Code_Test.py - The test colab mapping code for the interactive Toulouse Real Estate Map that skips the data cleaning.


## Metric
Submissions will be evaluated based on RMSE (root mean squared error). Lower the RMSE, better the model.


## Heroku Files
- Toulouse_Map_Code.py - Final executable python code from Toulouse_Map_Code.ipynb
- Procfile
- requirements.txt

## Prerequisites
Dependencies can be installed via:
pip install requirements.txt
geomap: pip install requirements_geomap.txt
modeling: pip install requirements_modeling.txt

## Possible Improvements
- Consolidate regularly the dataset by only scraping new listings that are not in the dataset yet
- Scrape from additional data sources. ie. Leboncoin
- Add a widget to the interactive map to be able to select per number of rooms/bedrooms 

## Author
Jeremy Ndeby - Creator - @jeremyndeby

