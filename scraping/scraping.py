#!/usr/bin/env python
# coding: utf-8



# ### Scraping with Beautiful Soup and Selenium ### #

# In[17]:


#import os
#import random
#import time

#import numpy as np
#import pandas as pd
#import seaborn as sns
#from bs4 import BeautifulSoup

#sns.set()

#from selenium import webdriver
#import sys

# In[19]:


pip freeze > requirements.txt


# In[2]:


chromedriver = "C:/Users/jerem/Anaconda3/envs/environment-deep-learning-cookbook/Lib/site-packages/notebook/tests/selenium/chromedriver_win32/chromedriver" # path to the chromedriver executable
chromedriver = os.path.expanduser(chromedriver)
print('chromedriver path: {}'.format(chromedriver))
sys.path.append(chromedriver)

driver = webdriver.Chrome(chromedriver)


# In[3]:


seloger_toulouse_url = 'https://www.seloger.com/immobilier/locations/immo-toulouse-31/bien-appartement/?LISTING-LISTpg='

def get_page_links(url, number_of_pages):
    page_links=[] # Create a list of pages links
    for i in range(1,number_of_pages+1):
        j = url + str(i)
        page_links.append(j)
    return page_links

#page_links = get_page_links(seloger_toulouse_url,3) 


# In[4]:


def get_appartment_links(pages, driver):
    
    # Setting a list of listings links
    appartment_links=[] 
    
    # Getting length of list 
    length = len(pages) 
    
    while len(pages) > 0: 
        for i in pages:
            
            #print('Extracting links from page',pages.index(i)+1,'out of',len(pages),'left')
            # we try to access a page with the new proxy
            try:
                driver.get(i)
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Extract links information via the find_all function of the soup object 
                listings = soup.find_all("a", attrs={"name": "classified-link"})
                page_data = [row['href'] for row in listings]
                appartment_links.append(page_data) # list of listings links
                
                pages.remove(i)

                print('There are',len(pages),'pages left to examine')
                time.sleep(random.randrange(11,21))
    
            except:
                print("Skipping. Connnection error")
                time.sleep(random.randrange(300,600))
                
    return appartment_links


# In[5]:


# Create a flatten function:
def flatten_list(appartment_links):
    appartment_links_flat=[]
    for sublist in appartment_links:
        for item in sublist:
            appartment_links_flat.append(item)
    return appartment_links_flat
        
#or appartment_links_flat = list(it.chain.from_iterable(appartment_links))

#appartment_links_flat = flatten_list(appartment_links)
# Check the number of links
#len(appartment_links_flat)


# In[6]:


def get_title(soup):
    try:
        title = soup.title.text
        return title
    except:
        return np.nan

    
def get_agency(soup):
    try:
        agency = soup.find_all("a", class_="agence-link")
        agency2 = agency[0].text
        agency3 = agency2.replace('\n', '').lower()
        return agency3
    except:
        return np.nan
    
    
def get_housing_type(soup):
    try:
        ht = soup.find_all("h2", class_="c-h2")
        ht2 = ht[0].text
        return ht2
    except:
        return np.nan
    
    
def get_city(soup):
    try:
        city = soup.find_all("p", class_="localite")
        city2 = city[0].text
        return city2
    except:
        return np.nan
    
    
def get_details(soup):
    try:
        details = soup.find_all("h1", class_="detail-title title1")
        details2 = details[0].text
        details3 = details2.replace('\n', '').lower()
        return details3
    except:
        return np.nan
    
    
def get_rent(soup):
    try:
        rent = soup.find_all("a", class_="js-smooth-scroll-link price")
        rent2 = rent[0].text
        rent3 = int(''.join(filter(str.isdigit, rent2)))
        return rent3
    except:
        return np.nan
    
def get_charges(soup):
    try:
        cha = soup.find_all("sup", class_="u-thin u-300 u-black-snow")
        cha2 = cha[0].text
        return cha2
    except:
        return np.nan
    
    
def get_rent_info(soup):
    try:
        rent_info = soup.find_all("section", class_="categorie with-padding-bottom")
        rent_info2 = rent_info[0].find_all("p", class_="sh-text-light")
        rent_info3 = rent_info2[0].text
        return rent_info3
    except:
        return 'None'
    
    
def get_criteria(soup):
    try:
        crit = soup.find_all("section", class_="categorie")
        crit2 = [div.text for ul in crit for div in ul.find_all("div", class_="u-left")]
        crit3=(" ; ".join(crit2)) # concatenate string items in a list into a single string
        return crit3
    except:
        return 'None'
    
    
def get_energy_rating(soup):
    try:
        ener = soup.find_all("div", class_="info-detail")
        ener2 = ener[0].text
        ener3 = int(''.join(filter(str.isdigit, ener2)))
        return ener3
    except:
        return np.nan
    
    
def get_gas_rating(soup):
    try:
        gas = soup.find_all("div", class_="info-detail")
        gas2 = gas[1].text
        gas3 = int(''.join(filter(str.isdigit, gas2)))
        return gas3
    except:
        return np.nan
    
    
def get_description(soup):
    try:
        descr = soup.find_all(class_="sh-text-light")
        descr2 = descr[0].text
        return descr2
    except:
        return 'None'


# In[7]:


def get_html_data(url, driver):
    driver.get(url)
    #time.sleep(random.lognormal(0,1))
    time.sleep(random.randrange(5,15))
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup


# In[8]:


def get_apartment_data(driver,links):
    apartment_data = []

    while len(links) > 0: 
        for i in links:

            print('Now extracting data from listing {} out of {}.'.format(links.index(i)+1,len(links)))

            # we try to access a page with the new proxy
            try:
                soup = get_html_data(i,driver)
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
                    time.sleep(random.randrange(300,600))

                # if access to the listing granted then extract data and remove the listing from the list
                else:
                    appartment_data.append([i,title,agency,housing_type,city,details,rent,charges,rent_info,
                                            criteria,energy_rating,gas_rating,description]) 
                    links.remove(i)
                    print('Good! There are {} listings left to examine.'.format(len(links)))

            except:
                print("Skipping. Connnection error")
                time.sleep(random.randrange(60,120))
     
    
    df = pd.DataFrame(appartment_data,columns = ['link','title','agency','housing_type','city','details','rent','charges','rent_info',
                                            'criteria','energy_rating','gas_rating','description'])
    
    return df


# In[9]:


page_links = get_page_links(seloger_toulouse_url,96) 
appartment_links = get_appartment_links(page_links,driver)
appartment_links_flat = flatten_list(appartment_links)
df_appartment = get_appartment_data(driver,appartment_links_flat)

 
# In[10]:


df_appartment.shape


# In[15]:


df_appartment.head()


# In[16]:


df_appartment.to_csv('data_seloger_scraping_part1.csv',index=False)


# Next steps for improvement:
# - Rotate the user agent
# - Rotate of proxies (proxy pool)
# - Only extract the new listings to consolidate our data: Trier par date => afin d’avoir les offres récentes (le but étant de ne pas scraper 2 fois la même annonce, donc si 200 nouvelles annonces ont été publiées depuis hier, le scrap d’aujourd’hui doit s’arrêter quand il aura scraper ces 200 annonces et qu’il retrouve par la suite une annonce qu’il a déjà scrapé la veille)

# #### References & code:
# - https://medium.com/france-school-of-ai/web-scraping-avec-python-apprenez-%C3%A0-utiliser-beautifulsoup-proxies-et-un-faux-user-agent-d7bfb66b6556
# - https://towardsdatascience.com/looking-for-a-house-build-a-web-scraper-to-help-you-5ab25badc83e
# - https://medium.com/@ben.sturm/scraping-house-listing-data-using-selenium-and-beautiful-soup-1cbb94ba9492

# In[ ]:




