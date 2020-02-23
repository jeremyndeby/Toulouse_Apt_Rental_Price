#!/usr/bin/env python
# coding: utf-8


# ### Data scraping workflow with Beautiful Soup and Selenium ### #

# Source: The code is based on the work of Ben Sturm available here: https://medium.com/@ben.sturm/scraping-house-listing-data-using-selenium-and-beautiful-soup-1cbb94ba9492
# Info: Here we modified its code to match the characteristics of the website SeLoger.com


# Import libraries
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

# pip freeze > requirements.txt


# Instantiate a WebDriver object
# We need to specify the chromedriver location specific to our computer.
chromedriver = "C:/Users/jerem/Anaconda3/envs/environment-deep-learning-cookbook/Lib/site-packages/notebook/tests/selenium/chromedriver_win32/chromedriver"
chromedriver = os.path.expanduser(chromedriver)
print('chromedriver path: {}'.format(chromedriver))
sys.path.append(chromedriver)
driver = webdriver.Chrome(chromedriver)

# Specify the URL of the main SeLoger.com homepage
seloger_toulouse_url = 'https://www.seloger.com/immobilier/locations/immo-toulouse-31/bien-appartement/?LISTING-LISTpg='


# Create a function that returns a list of URLs of all pages
def get_page_links(url, number_of_pages):
    page_links = []  # Create a list of pages links
    for i in range(1, number_of_pages + 1):
        j = url + str(i)
        page_links.append(j)
    return page_links


# Create a function that returns a list of all links from all the pages
def get_apartment_links(pages, driver):
    # Setting a list of listings links
    apartment_links = []

    # Getting length of list 
    length = len(pages)

    while len(pages) > 0:
        for i in pages:

            # print('Extracting links from page',pages.index(i)+1,'out of',len(pages),'left')
            # we try to access a page with the new proxy
            try:
                driver.get(i)
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Extract links information via the find_all function of the soup object 
                listings = soup.find_all("a", attrs={"name": "classified-link"})
                page_data = [row['href'] for row in listings]
                apartment_links.append(page_data)  # list of listings links

                pages.remove(i)

                print('There are', len(pages), 'pages left to examine')
                time.sleep(random.randrange(11, 21))

            except:
                print("Skipping. Connnection error")
                time.sleep(random.randrange(300, 600))

    return apartment_links


# Create a flatten function
def flatten_list(apartment_links):
    apartment_links_flat = []
    for sublist in apartment_links:
        for item in sublist:
            apartment_links_flat.append(item)
    return apartment_links_flat


# Create a function that returns the title of the listing
def get_title(soup):
    try:
        title = soup.title.text
        return title
    except:
        return np.nan


# Create a function that returns the agency of the listing
def get_agency(soup):
    try:
        agency = soup.find_all("a", class_="agence-link")
        agency2 = agency[0].text
        agency3 = agency2.replace('\n', '').lower()
        return agency3
    except:
        return np.nan


# Create a function that returns the housing type of the listing
def get_housing_type(soup):
    try:
        ht = soup.find_all("h2", class_="c-h2")
        ht2 = ht[0].text
        return ht2
    except:
        return np.nan


# Create a function that returns the city of the listing
def get_city(soup):
    try:
        city = soup.find_all("p", class_="localite")
        city2 = city[0].text
        return city2
    except:
        return np.nan


# Create a function that returns the details of the listing
def get_details(soup):
    try:
        details = soup.find_all("h1", class_="detail-title title1")
        details2 = details[0].text
        details3 = details2.replace('\n', '').lower()
        return details3
    except:
        return np.nan


# Create a function that returns the rental price of the listing
def get_rent(soup):
    try:
        rent = soup.find_all("a", class_="js-smooth-scroll-link price")
        rent2 = rent[0].text
        rent3 = int(''.join(filter(str.isdigit, rent2)))
        return rent3
    except:
        return np.nan


# Create a function that returns the type of charges of the listing
def get_charges(soup):
    try:
        cha = soup.find_all("sup", class_="u-thin u-300 u-black-snow")
        cha2 = cha[0].text
        return cha2
    except:
        return np.nan


# Create a function that returns the rental price additional information of the listing
def get_rent_info(soup):
    try:
        rent_info = soup.find_all("section", class_="categorie with-padding-bottom")
        rent_info2 = rent_info[0].find_all("p", class_="sh-text-light")
        rent_info3 = rent_info2[0].text
        return rent_info3
    except:
        return 'None'


# Create a function that returns all criteria of the listing
def get_criteria(soup):
    try:
        crit = soup.find_all("section", class_="categorie")
        crit2 = [div.text for ul in crit for div in ul.find_all("div", class_="u-left")]
        crit3 = (" ; ".join(crit2))  # concatenate string items in a list into a single string
        return crit3
    except:
        return 'None'


# Create a function that returns the energy rating of the listing
def get_energy_rating(soup):
    try:
        ener = soup.find_all("div", class_="info-detail")
        ener2 = ener[0].text
        ener3 = int(''.join(filter(str.isdigit, ener2)))
        return ener3
    except:
        return np.nan


# Create a function that returns the gas rating of the listing
def get_gas_rating(soup):
    try:
        gas = soup.find_all("div", class_="info-detail")
        gas2 = gas[1].text
        gas3 = int(''.join(filter(str.isdigit, gas2)))
        return gas3
    except:
        return np.nan


# Create a function that returns the full raw description of the listing
def get_description(soup):
    try:
        descr = soup.find_all(class_="sh-text-light")
        descr2 = descr[0].text
        return descr2
    except:
        return 'None'


# Create a function that gets the html data from the URL specified and returns it as a Beautiful Soup object
def get_html_data(url, driver):
    driver.get(url)

    # Wait for a few seconds
    # time.sleep(random.lognormal(0,1))
    time.sleep(random.randrange(5, 15))
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup


# Create a function that puts all the functionality of reading in the html data and scraping the relevant fields
# and returns a raw dataframe
def get_apartment_data(driver, links):
    apartment_data = []

    while len(links) > 0:
        for i in links:

            print('Now extracting data from listing {} out of {}.'.format(links.index(i) + 1, len(links)))

            # we try to access a page with the new proxy
            try:
                soup = get_html_data(i, driver)
                title = get_title(soup)
                agency = get_agency(soup)
                housing_type = get_housing_type(soup)
                city = get_city(soup)
                details = get_details(soup)
                rent = get_rent(soup)
                charges = get_charges(soup)
                rent_info = get_rent_info(soup)
                criteria = get_criteria(soup)
                energy_rating = get_energy_rating(soup)
                gas_rating = get_gas_rating(soup)
                description = get_description(soup)

                # if listings is not available anymore then remove the listing from the list
                if title == 'Location appartements Toulouse (31) | Louer appartements à Toulouse 31000':
                    print('This appartment is no longer available.')
                    links.remove(i)

                # if listing not accessible (robot) then go to the next one and try again later
                elif pd.isna(housing_type) == True and pd.isna(city) == True and pd.isna(rent) == True:
                    print('You Shall Not Pass!')
                    time.sleep(random.randrange(300, 600))

                # if access to the listing granted then extract data and remove the listing from the list
                else:
                    apartment_data.append([i, title, agency, housing_type, city, details, rent, charges, rent_info,
                                           criteria, energy_rating, gas_rating, description])
                    links.remove(i)
                    print('Good! There are {} listings left to examine.'.format(len(links)))

            except:
                print("Skipping. Connnection error")
                time.sleep(random.randrange(60, 120))

    df = pd.DataFrame(apartment_data, columns=['link', 'title', 'agency', 'housing_type', 'city',
                                               'details', 'rent', 'charges', 'rent_info', 'criteria',
                                               'energy_rating', 'gas_rating', 'description'])
    return df


# Call the functions
page_links = get_page_links(seloger_toulouse_url, 96)
apartment_links = get_apartment_links(page_links, driver)
apartment_links_flat = flatten_list(apartment_links)
df_apartment = get_apartment_data(driver, apartment_links_flat)

# Size of the dataset
print("Initial data size is: {} ".format(df_apartment.shape))


# ## Export the file
df_apartment.to_csv('data_seloger_raw.csv', index=False)
print("Data exported")


# ## Next steps for improvement:
# - Rotate the user agent
# - Rotate of proxies (proxy pool)
# - Only extract the new listings to consolidate our data
