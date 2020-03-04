# Toulouse_Apt_Rental_Price:

Under Construction

## Introduction: About the Project

Thousands of new students move to Toulouse (France) every year for their studies for a total. It is often difficult for them to know which neighborhoods are the cheapest or to identify which apartments have been undervalued/overvalued based on their set of features. However, many research and tutorials have been focusing on training models to predict property selling prices, but relatively few on predicting apartment rental prices. This project aims to fill the gap and provide useful insights to people, especially students, to help them in their decision-making as to which apartment to rent out.

In summary, the aims of this project are: 
1. To build a web scraper to collect the apartment listings in Toulouse from the platform SeLoger.com.
2. To build a model to estimate what should be the correct rental price given different features and their property.
3. To provide an interactive geographic map to get more familiar with the different neighborhoods of the city. 

## Predicting Rental Price
After the data has been scraped, it is a natural progression to explore and model the relationships between the property features and the rental prices. Seven different types of algorithms were used to train the models. We will focus on comparing the performance and results of the following seven algorithms:

1. Linear Regression
2. Lasso
3. Ridge
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. Extreme Gradient Boosting Regressor
7. Light Gradient Boosting Regressor

![Models Comparison](modeling/model_comparison.png)

### Metric
Submissions will be evaluated based on RMSE (root mean squared error). Lower the RMSE, better the model.
- R-squared: meassures the % of variance in the target variable explained by the data
- RMSE: measures the distance between the predicted values and actual values

## Interactive Map
The interactive geographical map below provides details on Toulouse apartments for rent currently available. 
The chart breaks down the apartments for rent by:
- Median Rental Price
- Average Rental Price
- Median Area in Square Meters
- Average Area in Square Meters
- Median Rental Price per Square Meter
- Average Rental Price per Square Meter
- Number of Apartments for Rent

[Link to Interactive Map of Toulouse Apartments For Rent 2020](https://toulouse-apt-rental-price.herokuapp.com/geomap)

[Capture of Geographical Map](geomap/capture_geomap.PNG)


## Data: A data directory
In Toulouse, SeLoger.com is an online marketplace allowing real estate agencies and owners to post listings on their website. The website gathers most of apartments for rent of the city. The data is based on listings from SeLoger.com and is collected using a custom scraper. 
- data_seloger_raw.csv - The raw data scraped from [SeLoger.com](https://www.seloger.com/immobilier/locations/immo-toulouse-31/bien-appartement/?LISTING-LISTpg=0) for all currently available apartments for rent in Toulouse, France.
- data_seloger_clean.csv - The cleaned data after processing and cleaning steps.
- data_geomap.csv - The data use for the geo mapping part.
- recensement-population-2015-grands-quartiers-population.geojson - The mapping data downloaded from [data.toulouse-metropole.fr](https://data.toulouse-metropole.fr/explore/dataset/recensement-population-2015-grands-quartiers-population/export/)
- data_model.csv - The data used for the modeling part


## Notebooks
There are three notebooks within this repository representing different stages of the project:
- scraper.py - Scrape the data from SeLoger.com
- data_processing.py - Prepare raw data scraped from 
- EDA.py - EDA of the cleaned data
- model.py - Build and compare several models and find the best hyper-parameters for the final model
- geomap.py - The mapping code for the interactive geographical map of Toulouse real sstate



## Heroku Files
- Procfile
- requirements.txt
- runtime.txt

## Prerequisites
Dependencies can be installed via:
pip install requirements.txt

## Possible Improvements
- Consolidate regularly the dataset by only scraping new listings that are not in the dataset yet
- Scrape from additional data sources. ie. Leboncoin
- Add a widget to the interactive map to be able to select per number of rooms/bedrooms 

## Author
Jeremy Ndeby - Creator - [@jeremyndeby](https://github.com/jeremyndeby)

If you have any feedback or questions for this project, feel free to contact me via my [LinkedIn](https://www.linkedin.com/in/jeremyndeby/)
