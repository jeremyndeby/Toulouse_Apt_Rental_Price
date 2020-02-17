# Toulouse_Apt_Rental_Price:

The objective of this project is to model the rental prices of appartments in Toulouse, France. The aim is to build a model to estimate what should be the correct  rental price given different features and their property.

## Introduction: About the Project
When moving to a new city it is often difficult to now which neighborhoods are the cheapest but also to make sure that the rental price have not been overvalued based on the set of feature of the apartment. In Toulouse, SeLoger.com is an online marketplace allowing real estate agencies and owners to post listings on their website. The website gathers most of apartments for rent of the city.

The goal of this project is double: 
1. To analyze the apartment listings of London and help people (especially new students in the city) estimate what the correct price of each apartment should be given the set of features. 
2. To provide an interactive geographic map to get more familiar with the different neighborhoods of the city. 

## data - a data directory
- tttt.csv - The raw data scraped from https://www.seloger.com/immobilier/locations/immo-toulouse-31/bien-appartement/?LISTING-LISTpg=0 for all currently available apartments for rent in Toulouse, France.
- data_seloger_EDA_part3.csv - The cleaned data after processing and cleaning steps.
- data_seloger_EDAforSpatial_part3.csv - The cleaned dataset to use for the mapping part.
- recensement-population-2015-grands-quartiers-population.geojson - The mapping data downloaded from https://data.toulouse-metropole.fr/explore/dataset/recensement-population-2015-grands-quartiers-population/export/

## Colab Notebooks
- Toulouse_Map_Code_Final.py - The colab mapping code for the interactive Toulouse Real Estate Map.
- Toulouse_Map_Code_Test.py - The test colab mapping code for the interactive Toulouse Real Estate Map that skips the data cleaning.

## Heroku Files
- Toulouse_Map_Code.py - Final executable python code from Toulouse_Map_Code.ipynb
- Procfile
- requirements.txt

## Prerequisites
Dependencies can be installed via:
pip install requirements.txt

## Author
Jeremy Ndeby - Creator - @jeremyndeby

