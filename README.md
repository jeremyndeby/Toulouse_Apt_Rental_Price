# Toulouse_Apt_Rental_Price
Prediction of Rental Prices and Interactive Geographic Map of apartments in Toulouse, France.


The contents include the following:

## data - a data directory
- tttt.csv - The raw data downloaded from https://www.seloger.com/immobilier/locations/immo-toulouse-31/bien-appartement/?LISTING-LISTpg=0 for all currently available apartments for rent in Toulouse, France.
- recensement-population-2015-grands-quartiers-population.geojson - The mapping data downloaded from https://data.toulouse-metropole.fr/explore/dataset/recensement-population-2015-grands-quartiers-population/export/
- 2018MedianHomes-Final.csv - The median home prices scraped from the NAR Median Sales Price of Existing SFH for MSAs. https://www.nar.realtor/sites/default/files/documents/metro-home-prices-q2-2019-single-family-2019-08-07.pdf
- RankedTechIncome.csv - The Bureau of Labor Statistics - Occupational Employment Statistics income data for 2018 by MSA. https://www.bls.gov/oes
- data_seloger_EDAforSpatial_part3.csv - The cleaned dataset to use for the mapping part.
- data_seloger_EDAforSpatial_part3.csv - A test dataset to use if you want to skip the data cleaning steps.

## Colab Notebooks
- Toulouse_Map_Code_Final.ipynb - The colab mapping code for the interactive Toulouse Real Estate Map.
- Toulouse_Map_Code_Test.ipynb - The test colab mapping code for the interactive Toulouse Real Estate Map that skips the data cleaning.

## Heroku Files
- Toulouse_Map_Code.py - Final executable python code from Toulouse_Map_Code.ipynb
- Procfile
- requirements.txt
