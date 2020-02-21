#!/usr/bin/env python
# coding: utf-8

# # II. Data Preparation (Pre-EDA)

# In[1]:


import pandas as pd
from IPython.display import display
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
from matplotlib.colors import LogNorm
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# Now that the scraping part is finished, data extracted still need some cleaning and processing before behing analyzed. 
# Indeed as we will see below, there are some features like 'rent_info', 'criteria' or 'description' that holds a lot of information that needs to be extracted.  

# ## 1. Data import
# Here we are importing the data from the csv file created in the previous part.

# In[2]:


# load data 
df = pd.read_csv('extract_appartment_seloger_202001_v1.csv')

# check the size of the dataset
df.shape


# In[3]:


type(df)


# ## 2. Quick inspection of the features
# There are some features that needs more processing than others. In order to know more about each of them we will inspect each feature one by one to get more familiar with our dataset.

# In[4]:


# first visual inspection of the dataset
df.head()


# In[5]:


#  columns in the dataset:
print(df.columns)


# ### Link

# In[6]:


# Describe
print(df['link'].describe())


# Here we can see that some listings were published several times (up to four times) over the different listing pages we extracted the data from. In the next part we will need to deduplicate our dataset based on this feature. 

# ### Title

# In[7]:


# Describe
print(df['title'].describe())


# In[8]:


# Check the values of the feature for the first i rows
for i in range(5):
    print(df.loc[i, 'title'])


# From the 'title' feature we can extract:
# - the number of rooms of the appartment
# - the city
# - the size of the appartment in square meters
# - the price of the rent

# ### Agency

# In[9]:


# Describe
print(df['agency'].describe())


# In[10]:


# Frequency of the value feature (pandas)
print(df['agency'].value_counts())


# There are 142 different real estate agencies in our dataset with SNG Extensia the agency with the highest number of appartment listed (before deduplication). There seems also to be a share of the appartment without an agency meaning that the landlord does not go through an agency to rent its appartment. 
# 
# This feature might play a double role on the rent as: 
# - going through a real estate agenxy or not might have an impact on the rent as agencies take a commission on each appartment rented
# - the size and reputation of the agency might have an impact on the rent as renowned agencies might take an higher comission on each appartment rented

# ### City

# In[11]:


# Describe
print(df['city'].describe())


# In[12]:


# Frequency Counts of the feature
print(df['city'].value_counts())


# We can see four different cities displayed but when taking a closer look at each unique value we can deduce there are all refering to Toulouse: 
# - 'Toulouse.': typo, same as Toulouse
# - '31000': One of Toulouse's postal code
# - 'Saint Martin du Touch': former autonomous village that has become a neighborhood of Toulouse. It is part of sub-division 6.1 of sector 6 of the city.
# 
# This is not surprising as our goal was to extract data from appartment in Toulouse only. This feature will then probably be dropped later.

# ### Housing type

# In[13]:


# Describe
print(df['housing_type'].describe())


# There is only one housing type in the dataset. This is not surprising as our goal was to extract data from appartment listings only. This feature will probably be dropped later. 

# ### Details

# In[14]:


# Describe
print(df['details'].describe())


# In[15]:


# Check the values of the feature for the first i rows
for i in range(5):
    print(df.loc[i, 'details'])


# From the 'details' feature we can extract:
# - the size of the appartment in square meters
# - the neighborhood/sector of the appartment

# ### Rent

# In[16]:


# Describe
print(df['rent'].describe())


# The rent feature does not need any transformation. 

# ### Charges

# In[17]:


# Frequency Counts of ta feature (pandas)
print(df['charges'].value_counts())


# Only two listings does not include charges in the rent (before deduplication). 

# ### Rent information

# In[18]:


# Describe
print(df['rent_info'].describe())


# In[19]:


# Check the values of the feature for the first i rows
for i in range(3):
    print(df.loc[i, 'rent_info'])


# Most of the listings do not have any rent information but when available we can extract: 
# - The renting provisions amount for the appartement
# - The renting fees amount for the appartement
# - The deposit amount for the appartement

# ### Criteria 

# In[20]:


# Describe
print(df['criteria'].describe())


# In[21]:


# Check the values of the feature for the first i rows
for i in range(3):
    print(df.loc[i, 'criteria'])


# From this feature we will extract quite a few information from different type such as: 
# - Inside features
# - Outside features
# - Advantages 
# - Neighborhood/Sector features

# ### Energy Rating

# In[22]:


# Describe
print(df['energy_rating'].describe())


# The energy rating feature does not need any transformation.

# ### Gas Rating

# In[23]:


# Describe
print(df['gas_rating'].describe())


# The gas rating feature does not need any transformation.

# ### Description

# In[24]:


# Describe
print(df['description'].describe())


# In[25]:


# Check the values of the feature for the first i rows
for i in range(3):
    print(df.loc[i, 'description'])


# From this feature we will extract quite a few information from different type such as: 
# - Inside features
# - Outside features
# - Advantages 
# - Neighborhood/Sector features

# ## 3. Data transformation
# After inspection we determined that various new features could be created from the 'details', 'rent_info', 'criteria' and 'description' features. 
# To facilitate the extraction of the information from those features we will for each feature of interest:
# - return the lowercased strings so we can capture every variant of a same word or sentence
# - remove accents from each string as we know that there are lots of accents in french but there are not always used the same way depending on the person
# - replace '-' characters from each string with an emtpy space so we can capture every variant of a same composed word

# In[26]:


import unidecode

# Apply some transformation to the features 'details', 'rent_info', 'criteria' and 'description'
for columns in ['details','rent_info','criteria','description']:
    df[columns] = [unidecode.unidecode(i) for i in df[columns]]
    df[columns] = df[columns].str.replace('-', ' ')
    #df[columns] = df[columns].str.replace(',', '')
    df[columns] = df[columns].str.lower() 


# In[27]:


df.head()


# Now we can start to create some new clean and usable features based on the existing 'details', 'rent_info', 'criteria' and 'description' raw features:

# ### About the renting price
# Let's extract, when available, information on the renting price such as:
# - The renting provisions amount for the appartement
# - The renting fees amount for the appartement
# - The deposit amount for the appartement

# In[28]:


# Create a renting provisions amount feature
df['provisions'] = df['rent_info'].str.extract('provisions pour charges avec regularisation annuelle [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
df['provisions'] = df['provisions'].str.replace(',', '')

# Create a renting fees feature
df['fees'] = df['rent_info'].str.extract('honoraires ttc a la charge du locataire [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
df['fees'] = df['fees'].str.replace(',', '')

# Create a deposit amount feature
df['deposit'] = df['rent_info'].str.extract('depot de garantie [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
df['deposit'] = df['deposit'].str.replace(',', '')


# We need then to extract all the hidden information from 'details', 'criteria' and 'description' features. 
# We determined which information to extract based on some visual inspection of the listings but also based on experience.
# 
# We also noticed that sometimes some specific information can be spread between those three features without any logic pattern:
# - sometimes the description of the appartment is more complete than the criteria listed or provide additional information
# - sometimes some information can only be find in 'criteria' and not in the description of the appartment
# - concerning the neighborhood of the appartment, the description of the appartment give more precision on the location compared to 'details', but it happens that, for some appartment, there is no mention of the neighborhood in the description and thus we have to trust the feature 'details'.
# 
# 
# Those issues are probably due to a lack of consistency in the filling of the listings. Inorder to tackle them we will then makes sure to extract new features from all three features when needed.

# ### Inside of the appartment features
# 
# Let's extract, when available, information related to the inside of the appartment such as:
# - The total size of the appartment in square meters
# - The total number of rooms
# - If the appartment has an entrance area or not
# - If the appartment is a duplex or not
# 
# - If the appartment has a living room or not
# - The total size of the appartment in square meters
# 
# 
# - If the appartment has or not an equipped kitchen
# - If the appartment has or not an open-plan kitchen
# 
# 
# - The total number of bedrooms
# 
# 
# - The total number of toilets
# - If the appartment has separate toilets or not
# 
# 
# - The total number of balconies
# - The total number of terraces
# 
# 
# - If the appartment has a wooden floor or not
# - If the appartment has a fireplace or not
# - If the appartment has some storage inside
# 
# 
# - The heating system of the appartment

# In[29]:


# Max function: return the max between 2 features and drop the features used
def max_func(df,feat_criteria,feat_description):
    feat = df[[feat_criteria, feat_description]].max(axis=1)
    df.drop([feat_criteria,feat_description], axis=1, inplace=True) 
    return feat


### APPARTMENT
# Create a total area size in square meter feature
df['area'] = df['criteria'].str.extract('surface de ([\d]{0,},?[\d]{1,}) m2')
df['area'] = df['area'].str.replace(',', '.')

# Create a total number of rooms feature
df['rooms'] = df['criteria'].str.extract('([\d]{1,}) piece')
df["rooms"] = df["rooms"].fillna('0') # if nan then 0

# Create an entrance feature (dummy variable)
df['entrance'] = df['criteria'].str.contains('; entree ;', regex=True).astype(int)

# Create a duplex feature (dummy variable)
df['duplex_crit'] = df['criteria'].str.contains('; duplex ;', regex=True).astype(int)
df['duplex_descr'] = df['description'].str.contains('; duplex ;', regex=True).astype(int)
df['duplex'] = max_func(df,'duplex_crit','duplex_descr')


### LIVING ROOM
# Create a living room feature (dummy variable)
df['livingroom'] = df['criteria'].str.contains('; sejour ;', regex=True).astype(int)

# Create a total living room area size in square meter feature
df['livingroom_area'] = df['criteria'].str.extract('sejour de ([\d]{0,},?[\d]{1,}) m2')
df["livingroom_area"] = df["livingroom_area"].fillna('0') # if nan then 0


### KITCHEN
# Create an equipped kitchen feature (dummy variable)
df['equipped_kitchen_crit'] = df['criteria'].str.contains('cuisine equipe|cuisine americaine equipe', regex=True).astype(int)
df['equipped_kitchen_descr'] = df['description'].str.contains('cuisine equipe|cuisine américaine equipe', regex=True).astype(int)
df['equipped_kitchen'] = max_func(df,'equipped_kitchen_crit','equipped_kitchen_descr')


# Create an open-plan kitchen feature (dummy variable)
df['openplan_kitchen_crit'] = df['criteria'].str.contains('cuisine americaine', regex=True).astype(int)
df['openplan_kitchen_descr'] = df['description'].str.contains('cuisine americaine', regex=True).astype(int)
df['openplan_kitchen'] = max_func(df,'openplan_kitchen_crit','openplan_kitchen_descr')


### BEDROOMS
# Create a total number of bedrooms feature
df['bedrooms'] = df['criteria'].str.extract('([\d]{1,}) chambre')
df["bedrooms"] = df["bedrooms"].fillna('0') # if nan then 0

### BATHROOMS
# Create a total number of bathrooms feature
df['bathrooms'] = df['criteria'].str.extract('([\d]{1,}) salle de bain')
df["bathrooms"] = df["bathrooms"].fillna('0') # if nan then 0

# Create a total number of shower rooms feature
df['shower_rooms'] = df['criteria'].str.extract('([\d]{1,}) salle d\'eau')
df["shower_rooms"] = df["shower_rooms"].fillna('0') # if nan then 0


### TOILETS
# Create a total number of toilets feature
df['toilets'] = df['criteria'].str.extract('([\d]{1,}) toilette')

# Create a separate toilet feature (dummy variable)
df['separate_toilet_crit'] = df['criteria'].str.contains('toilettes separe', regex=True).astype(int)
df['separate_toilet_descr'] = df['description'].str.contains('toilettes separe', regex=True).astype(int)
df['separate_toilet'] = max_func(df,'separate_toilet_crit','separate_toilet_descr')


### BALCONY, TERRACES
# Create a total number of balconies feature
df['balcony'] = df['criteria'].str.extract('([\d]{1,}) balcon')
df["balcony"] = df["balcony"].fillna('0') # if nan then 0

# Create a total number of terraces feature
df['terraces'] = df['criteria'].str.extract('([\d]{1,}) terrasse')
df["terraces"] = df["terraces"].fillna('0') # if nan then 0


### WOODEN FLOOR, FIREPLACE, INSIDE STORAGE
# Create a wooden floor feature (dummy variable)
df['wooden_floor'] = df['criteria'].str.contains('parquet', regex=True).astype(int)

# Create a fireplace feature (dummy variable)
df['fireplace_crit'] = df['criteria'].str.contains('cheminee', regex=True).astype(int)
df['fireplace_descr'] = df['description'].str.contains('cheminee', regex=True).astype(int)
df['fireplace'] = max_func(df,'fireplace_crit','fireplace_descr')


# Create an inside storage feature (dummy variable)
df['storage'] = df['criteria'].str.contains('rangement', regex=True).astype(int)


### HEATING
# Create an heating function
def  heating_func(df):
    df['heating']=np.nan
    if 'chauffage gaz collectif' in df['criteria'] or 'chauffage au gaz collectif' in df['criteria'] or 'chauffage gaz collectif' in df['description'] or 'chauffage au gaz collectif' in df['description']:
        return 'collective_gas'
    elif 'chauffage individuel gaz' in df['criteria'] or 'chauffage individuel au gaz' in df['criteria'] or 'chauffage individuel gaz' in df['description'] or 'chauffage individuel au gaz' in df['description']:
        return 'individual_gas'
    elif 'chauffage gaz' in df['criteria'] or 'chauffage au gaz' in df['criteria'] or 'chauffage gaz' in df['description']  or 'chauffage au gaz' in df['description']:
        return 'gas'    
    elif 'chauffage individuel electrique' in df['criteria'] or 'chauffage electrique' in df['criteria'] or 'chauffage individuel electrique' in df['description'] or 'chauffage electrique' in df['description']:
        return 'individual_electrical'
    elif 'chauffage individuel' in df['criteria'] or 'chauffage individuel' in df['description']:
        return 'individual' 
    elif 'chauffage central' in df['criteria'] or 'chauffage central' in df['description']:
        return 'central' 
    elif 'chauffage sol' in df['criteria'] or 'chauffage au sol' in df['criteria'] or 'chauffage sol' in df['description'] or 'chauffage au sol' in df['description']:
        return 'central' 
    else:
        return df['heating']

    
df['heating'] = df.apply(heating_func, axis=1)


# ### Outside features
# Let's extract, when available, information related to the outside of the appartment such as:
# - The number of parking places
# - If the appartment has a cellar or not
# - If the building has a pool or not
# - The year of construction of the building
# - The total number of floors of the building
# - The floor number of the appartment

# In[30]:


# Create a parking feature
df['parking'] = df['criteria'].str.extract('([\d]{1,}) parking')
df["parking"] = df["parking"].fillna('0') # if nan then 0

# Create a cellar feature (dummy variable)
df['cellar'] = df['criteria'].str.contains('; cave ;', regex=True).astype(int)

# Create a pool feature (dummy variable)
df['pool_crit'] = df['criteria'].str.contains('piscine', regex=True).astype(int)
df['pool_descr'] = df['description'].str.contains('piscine', regex=True).astype(int)
df['pool'] = max_func(df,'pool_crit','pool_descr')

# Create a year of construction of the building feature
df['construction_year'] = df['criteria'].str.extract('annee de construction ([\d]{4,4})')

# Create the total number of floors of the building feature
df['bldg_flr_nb'] = df['criteria'].str.extract('batiment de ([\d]{1,}) etage')

# Create the floor number of the appartment feature
df['apt_flr_nb'] = df['criteria'].str.extract('; au ([\d]{1,})+(?:er|eme) etage')
df.loc[ df['criteria'].str.contains('au rez-de-chausse', case=False), 'apt_flr_nb'] = '0'


# ### Advantages
# Let's extract, when available, advantages of the appartment such as:
# - If the appartment is furnished or not
# - If the appartment has been fully renovated or not
# 
# 
# - If the building has an elevator or not 
# - If the appartment has an intercom or not
# - If the appartment has an digital code or not
# 
# 
# - The orientation of the appartment
# - If the appartment has a 'nice' view or not
# 
# 
# - If the building has a caretaker
# - If the appartment is adapted to persons with reduced mobility 

# In[31]:


### FURNISHED, RENOVATED
# Create a furnished feature (dummy variable)
df['furnished'] = df['criteria'].str.contains('; meuble ;', regex=True).astype(int)

# Create a renovated feature (dummy variable)
df['renovated'] = df['criteria'].str.contains('refait a neuf', regex=True).astype(int)


### ELEVATOR, INTERCOM, DIGITAL CODE
# Create an elevator feature (dummy variable)
df['elevator'] = df['criteria'].str.contains('ascenseur', regex=True).astype(int)

# Create an intercom feature (dummy variable)
df['intercom'] = df['criteria'].str.contains('interphone', regex=True).astype(int)

# Create a digital code feature (dummy variable)
df['digital_code'] = df['criteria'].str.contains('digicode', regex=True).astype(int)


### ORIENTATION, VIEW
# Create an orientation function
def  orientation_func(df):
    df['orientation']=np.nan
    if 'orientation nord, est' in df['criteria'] or 'orientation est, nord' in df['criteria']:
        return 'north_est'
    elif 'orientation nord, ouest' in df['criteria'] or 'orientation ouest, nord' in df['criteria']:
        return 'north_west'
    elif 'orientation sud, est' in df['criteria'] or 'orientation est, sud' in df['criteria']:
        return 'south_est'
    elif 'orientation sud, ouest' in df['criteria'] or 'orientation ouest, sud' in df['criteria']:
        return 'south_west' 
    
    elif 'orientation nord' in df['criteria']:
        return 'north'
    elif 'orientation sud' in df['criteria']:
        return 'south'
    elif 'orientation est' in df['criteria']:
        return 'est'
    elif 'orientation ouest'in df['criteria']:
        return 'west'
    else:
        return df['orientation']
    
df['orientation'] = df.apply(orientation_func, axis=1)

# Create a view feature (dummy variable)
df['view'] = df['criteria'].str.contains('; vue ;', regex=True).astype(int)


### CARETAKER, REDUCED-MOBILITY PERSONS
# Create a caretaker feature (dummy variable)
df['caretaker'] = df['criteria'].str.contains('; gardien ;', regex=True).astype(int)

# Create a reduced mobility feature (dummy variable)
df['reduced_mobility'] = df['criteria'].str.contains('adapte pmr', regex=True).astype(int)


# ### Neighborhood information
# 
# Due to the lack of consistency of the descriptions we need to use the websites https://www.toulouse.fr/vos-quartiers and https://fr.wikipedia.org/wiki/Quartiers_de_Toulouse to help us create the right neighborhoods and sectors. We learn from those websites that Toulouse is divided in 6 different sectors and each one of them is also divided in 3 to 4 neighborhoods for a total of 20 different neighborhoods.
# 
# Let's extract, when available, information on the neighborhood of the appartment such as:
# - The name of the neighborhood of the appartment
# - The name of the sector of the appartment: Each sector regroups 1 to 8 neighbourhoods
# - The postal code of the appartment
# - If the appartment is close to a metro station or not
# - If the appartment is close to a tram station or not
# - If the appartment is close to a bus station or not

# We first will create some dictionaries to enable us to retrieve easily neighborhood and sector codes and names:

# In[32]:


# Create a dictionary to change the neighborhood codes to neighborhood names.
neighborhood_dict = {'n1_1': 'Capitole - Arnaud Bernard - Carmes', 
                     'n1_2': 'Amidonniers - Compans Caffarelli', 
                     'n1_3': 'Les Chalets/Bayard/Belfort  Saint-Aubin/Dupuy', 
                     'n2_1': 'Saint-Cyprien',
                     'n2_2': 'Croix de Pierre - Route d\'Espagne',
                     'n2_3': 'Fontaine-Lestang - Arènes -Bagatelle - Papus - Tabar - Bordelongue - Mermoz - La Faourette',
                     'n2_4': 'Casselardit - Fontaine-Bayonne - Cartoucherie',
                     'n3_1': 'Minimes - Barrière de Paris - Ponts-Jumeaux',
                     'n3_2': 'Sept Deniers - Ginestous - Lalande',
                     'n3_3': 'Trois Cocus - Borderouge - Croix Daurade - Paleficat - Grand Selve',
                     'n4_1': 'Lapujade - Bonnefoy - Périole - Marengo - La Colonne',
                     'n4_2': 'Jolimont - Soupetard - Roseraie - Gloire - Gramont- Amouroux',
                     'n4_3': 'Bonhoure - Guilheméry - Château de l\'Hers - Limayrac - Côte Pavée',
                     'n5_1': 'Pont des Demoiselles - Ormeau - Montaudran - La Terrasse - Malepère',
                     'n5_2': 'Rangueil - Saouzelong - Pech David - Pouvourville',
                     'n5_3': 'Saint Michel - Le Busca - Empalot - Saint Agne',
                     'n6_1': 'Arènes Romaines - Saint Martin du Touch- Purpan',
                     'n6_2': 'Lardenne - Pradettes - Basso Cambo',
                     'n6_3': 'Mirail- Université - Reynerie - Bellefontaine',
                     'n6_4': 'Saint Simon - Lafourguette - Oncopole'}


# Create a dictionary to change the neighborhood codes to sector codes.
neighborhood_sector_dict = { 'n1_1': 'sector1', 
                             'n1_2': 'sector1', 
                             'n1_3': 'sector1', 
                             'n2_1': 'sector2',
                             'n2_2': 'sector2',
                             'n2_3': 'sector2',
                             'n2_4': 'sector2',
                             'n3_1': 'sector3',
                             'n3_2': 'sector3',
                             'n3_3': 'sector3',
                             'n4_1': 'sector4',
                             'n4_2': 'sector4',
                             'n4_3': 'sector4',
                             'n5_1': 'sector5',
                             'n5_2': 'sector5',
                             'n5_3': 'sector5',
                             'n6_1': 'sector6',
                             'n6_2': 'sector6',
                             'n6_3': 'sector6',
                             'n6_4': 'sector6'}

# Create a dictionary to change the sector codes to sector names.
sector_dict = {  'sector1': 'TOULOUSE CENTRE', 
                 'sector2': 'TOULOUSE RIVE GAUCHE', 
                 'sector3': 'TOULOUSE NORD', 
                 'sector4': 'TOULOUSE EST',
                 'sector5': 'TOULOUSE SUD EST',
                 'sector6': 'TOULOUSE OUEST'}


# Here we assign for each neighborhood some keywords that we will looking for in the features 'description' and 'details':

# In[33]:


### NEIGHBORHOOD

# Create the neighborhoods keywords lists
## Secteur 1: TOULOUSE CENTRE
#1.1 : Capitole - Arnaud Bernard - Carmes
#1.2 : Amidonniers - Compans Caffarelli
#1.3 : Les Chalets/Bayard/Belfort  Saint-Aubin/Dupuy

n1_1 = ['capitole','bernard','carmes',
                   'jacobins',' rome','dalbade','filatiers','sernin','taur',
                   'valade','occitane','wilson','baragnon','ozenne','trinite','esquirol',
                   'georges','fac de droit','jaures','jeanne d\'arc',
                   'st pierre','saint pierre','francois verdier','la daurade','lascrosses',
                  'saint etienne','st etienne','alsace lorraine','pont neuf','alsace','lorraine',
                  'place de la bourse','toulouse centre','hyper centre','hypercentre']

n1_2 = ['amidonniers','compans','caffarelli',
                   'bazacle','chapou','heracles','sébastopol','barcelone','brienne']

n1_3 = ['chalets','bayard','belfort','aubin','dupuy',
                   'concorde','raymond iv','belfort','gabriel peri','colombette','grand rond',
                  'honore serres','matabiau']


## Secteur 2: TOULOUSE RIVE GAUCHE
#2.1 : Saint-Cyprien
#2.2 : Croix de Pierre - Route d'Espagne
#2.3 : Fontaine-Lestang - Arènes -Bagatelle - Papus - Tabar - Bordelongue - Mermoz - La Faourette
#2.4 : Casselardit - Fontaine-Bayonne - Cartoucherie

n2_1 = ['cyprien',
                 'bourrassol','la grave','ravelin','roguet',' lucie','teinturiers','patte d\'oie','patte oie',
                  'fer a cheval','abattoirs','toulouse rive gauche']

n2_2 = ['croix de pierre','route d\'espagne',
                 'becanne','la digue','la pointe','avenue de muret','oustalous','deodat de severac']

n2_3 = ['lestang','arenes','bagatelle',' papus','tabar','bordelongue','mermoz','farouette',
                 'arenes','bigorre','lambert',' loire','morvan',' tellier',' touraine','vestrepain','hippodrome']

n2_4 = ['casselardit','fontaine-bayonne','fontaine bayonne','cartoucherie',
                 'barrière de bayonne','biarritz','les fontaines','zenith']


## Secteur 3: TOULOUSE NORD
#3.1 : Minimes - Barrière de Paris - Ponts-Jumeaux
#3.2 : Sept Deniers - Ginestous - Lalande
#3.3 : Trois Cocus - Borderouge - Croix Daurade - Paleficat - Grand Selve

n3_1 = ['minimes','barriere de paris','jumeaux',
                  'la vache','canal du midi']

n3_2 = ['deniers','ginestous','lalande',
                  'route de launaguet']

n3_3 = ['trois cocus','3 cocus','borderouge','croix daurade','paleficat','grand selve',
                  'izards']


## Secteur 4: TOULOUSE EST
#4.1 : Lapujade - Bonnefoy - Périole - Marengo - La Colonne
#4.2 : Jolimont - Soupetard - Roseraie - Gloire - Gramont- Amouroux
#4.3 : Bonhoure - Guilheméry - Château de l'Hers - Limayrac - Côte Pavée

n4_1 = ['lapujade','bonnefoy','periole','marengo','la colonne']

n4_2 = ['jolimont','soupetard','roseraie','gloire','gramont','amouroux',
        'argoulets']

n4_3 = ['bonhoure','guilhemery','l\'hers','limayrac','cote pavee',
                  'camille pujol','hers','grande plaine']


## Secteur 5: TOULOUSE SUD EST
#5.1 : Pont des Demoiselles - Ormeau - Montaudran - La Terrasse - Malepère
#5.2 : Rangueil - Saouzelong - Pech David - Pouvourville
#5.3 :Saint Michel - Le Busca - Empalot - Saint Agne

n5_1 = ['demoiselles','ormeau','montaudran','la terrasse ','malepere',
                  'exupery']

n5_2 = ['rangueil','saouzelong','pech david','pouvourville',
                   'faculte de pharmacie','paul sabatier','ramonville','rangeuil','palays','jules julien']

n5_3 = ['michel','busca','empalot',' agne',
                  'palais de justice','des plantes','ramier']


## Secteur 6: TOULOUSE OUEST
#6.1 : Arènes Romaines - Saint Martin du Touch- Purpan
#6.2 : Lardenne - Pradettes - Basso Cambo
#6.3 : Mirail- Université - Reynerie - Bellefontaine
#6.4 : Saint Simon - Lafourguette - Oncopole 
    
    
n6_1 = ['arenes romaines','martin ','du touch','purpan']

n6_2 = ['lardenne','pradettes','basso',
                  'bordeblanche','cepiere']

n6_3 = ['mirail','reynerie','bellefontaine']

n6_4 = ['simon','lafourguette','oncopole',
                  'ramee']


# Let's now create a function to search for the keywords we identified above and return the neighborhood code:

# In[34]:


# Create a neighborhood function for the 'description' feature
def  neighborhood_description_func(df):
    # n1_n
    for i in n1_1:
        if i in df['description']:
            return 'n1_1'
    for i in n1_2:
        if i in df['description']:
            return 'n1_2'
    for i in n1_3:
        if i in df['description']:
            return 'n1_3'  

    # n2_n
    for i in n2_1:
        if i in df['description']:
            return 'n2_1'
    for i in n2_2:
        if i in df['description']:
            return 'n2_2'
    for i in n2_3:
        if i in df['description']:
            return 'n2_3'
    for i in n2_4:
        if i in df['description']:
            return 'n2_4'

    # n3_n
    for i in n3_1:
        if i in df['description']:
            return 'n3_1'
    for i in n3_2:
        if i in df['description']:
            return 'n3_2'
    for i in n3_3:
        if i in df['description']:
            return 'n3_3'  
        
    # n4_n        
    for i in n4_1:
        if i in df['description']:
            return 'n4_1'
    for i in n4_2:
        if i in df['description']:
            return 'n4_2'
    for i in n4_3:
        if i in df['description']:
            return 'n4_3'  

    # n5_n
    for i in n5_1:
        if i in df['description']:
            return 'n5_1'
    for i in n5_2:
        if i in df['description']:
            return 'n5_2'
    for i in n5_3:
        if i in df['description']:
            return 'n5_3'  

    # n6_n
    for i in n6_1:
        if i in df['description']:
            return 'n6_1'
    for i in n6_2:
        if i in df['description']:
            return 'n6_2'
    for i in n6_3:
        if i in df['description']:
            return 'n6_3'  
    for i in n6_4:
        if i in df['description']:
            return 'n6_4'

# Create a neighborhood function for the 'details' feature
def  neighborhood_details_func(df):
    # n1_n
    for i in n1_1:
        if i in df['details']:
            return 'n1_1'
    for i in n1_2:
        if i in df['details']:
            return 'n1_2'
    for i in n1_3:
        if i in df['details']:
            return 'n1_3'  

    # n2_n
    for i in n2_1:
        if i in df['details']:
            return 'n2_1'
    for i in n2_2:
        if i in df['details']:
            return 'n2_2'
    for i in n2_3:
        if i in df['details']:
            return 'n2_3'
    for i in n2_4:
        if i in df['details']:
            return 'n2_4'

    # n3_n
    for i in n3_1:
        if i in df['details']:
            return 'n3_1'
    for i in n3_2:
        if i in df['details']:
            return 'n3_2'
    for i in n3_3:
        if i in df['details']:
            return 'n3_3'  
        
    # n4_n        
    for i in n4_1:
        if i in df['details']:
            return 'n4_1'
    for i in n4_2:
        if i in df['details']:
            return 'n4_2'
    for i in n4_3:
        if i in df['details']:
            return 'n4_3'  

    # n5_n
    for i in n5_1:
        if i in df['details']:
            return 'n5_1'
    for i in n5_2:
        if i in df['details']:
            return 'n5_2'
    for i in n5_3:
        if i in df['details']:
            return 'n5_3'  

    # n6_n
    for i in n6_1:
        if i in df['details']:
            return 'n6_1'
    for i in n6_2:
        if i in df['details']:
            return 'n6_2'
    for i in n6_3:
        if i in df['details']:
            return 'n6_3'  
    for i in n6_4:
        if i in df['details']:
            return 'n6_4'
        
df['nbhd_no_description'] = df.apply(neighborhood_description_func, axis=1)
df['nbhd_no_details'] = df.apply(neighborhood_details_func, axis=1) 

# Create a unique neighborhood feature based on the remark made previously
df['nbhd_no'] = df['nbhd_no_description']
df.loc[df.nbhd_no_description.isnull(), 'nbhd_no'] = df.loc[df.nbhd_no_description.isnull(), 'nbhd_no_details'] 

# Drop some features
df.drop(['nbhd_no_description','nbhd_no_details'], axis=1, inplace=True)


# Now we use the dictionaries create before to create a neighborhood name, sector code and sector name from the neighborhood code:

# In[35]:


# Create a neighborhood name from the dictionary neighborhood_dict
df['nbhd_name'] = df['nbhd_no'].map(neighborhood_dict)

# Create a sector code from the dictionary neighborhood_dict
df['sector_no'] = df['nbhd_no'].map(neighborhood_sector_dict)

# Create a sector name from the dictionary neighborhood_dict
df['sector_name'] = df['sector_no'].map(sector_dict)


# Check the frequency
#df.groupby(["sector_no", "nbhd_no"]).size()


df.head()


# In[36]:


# Create a postal code feature
df['postal_code'] = df['description'].str.extract('(31[\d]{3,3})')


### TRANSPORTATION
# Create a metro feature (dummy variable)
df['metro'] = df['description'].str.contains('metro', regex=True).astype(int)

# Create a metro feature (dummy variable)
df['tram'] = df['description'].str.contains('tram|tramway', regex=True).astype(int)

# Create a metro feature (dummy variable)
df['bus'] = df['description'].str.contains('bus|bus.|bus,', regex=True).astype(int)


# Let's now reorder our columns and drop the columns that are of no use for our study. 

# In[37]:


# Drop some columns
df.drop(['title','housing_type','city','details','rent_info','criteria','description'], axis=1, inplace=True)

# Reorder columns
df = df[['link', 'agency', 'postal_code', 'sector_no', 'sector_name', 'nbhd_no', 'nbhd_name',
         'rent', 'charges','provisions', 'fees', 'deposit',
         'energy_rating', 'gas_rating',
         'area', 'rooms', 'entrance', 'duplex',
         'livingroom', 'livingroom_area', 'equipped_kitchen', 'openplan_kitchen',
         'bedrooms', 'bathrooms', 'shower_rooms', 'toilets', 'separate_toilet',
         'balcony', 'terraces', 'wooden_floor', 'fireplace', 'storage', 'heating', 
         'parking', 'cellar', 'pool', 
         'construction_year', 'bldg_flr_nb', 'apt_flr_nb', 
         'furnished', 'renovated', 'elevator', 'intercom', 'digital_code', 'orientation', 'view', 'caretaker', 'reduced_mobility',  
         'metro', 'tram', 'bus']]


# We also check data type with .info()

# In[38]:


df.info()


# We convert to numeric columns that can be converted:

# In[39]:


# using apply method 
df[[ 'rent', 'provisions', 'fees', 'deposit',
     'energy_rating', 'gas_rating',
     'area', 'rooms', 'entrance', 'duplex',
     'livingroom', 'livingroom_area', 'equipped_kitchen', 'openplan_kitchen',
     'bedrooms', 'bathrooms', 'shower_rooms', 'toilets', 'separate_toilet',
     'balcony', 'terraces', 'wooden_floor', 'fireplace', 'storage', 
     'parking', 'cellar', 'pool', 
     'construction_year', 'bldg_flr_nb', 'apt_flr_nb', 
     'furnished', 'renovated', 'elevator', 'intercom', 'digital_code', 'view', 'caretaker', 'reduced_mobility',  
     'metro', 'tram', 'bus']] = df[['rent', 'provisions', 'fees', 'deposit',
     'energy_rating', 'gas_rating',
     'area', 'rooms', 'entrance', 'duplex',
     'livingroom', 'livingroom_area', 'equipped_kitchen', 'openplan_kitchen',
     'bedrooms', 'bathrooms', 'shower_rooms', 'toilets', 'separate_toilet',
     'balcony', 'terraces', 'wooden_floor', 'fireplace', 'storage', 
     'parking', 'cellar', 'pool', 
     'construction_year', 'bldg_flr_nb', 'apt_flr_nb', 
     'furnished', 'renovated', 'elevator', 'intercom', 'digital_code', 'view', 'caretaker', 'reduced_mobility',  
     'metro', 'tram', 'bus']].apply(pd.to_numeric) 


# ## 4. Verifications
# The DataFrame now looks a lot cleaner, but we still want to make sure it’s really usable before we start our analysis. 
# We use .describe() for this:

# In[40]:


df.describe()


# We can deduce from the above image that the work is not finished yet. Indeed we observe the presence of some issues: 
# - the highest energy rating indicated is above the legal maximum.
# - the highest gas rating indicated is above the legal maximum.
# - the maximum number of toilets is 10. 
# - the maximum number of terraces places is 21. 
# - the maximum number of parking places is 29. 
# 
# In view of these inconsistencies, we can  can imagine that we are dealing here with mistakes from the agency or typos which we confirmed by checking the online descriptions of these appartments.
# 
# We decide to drop these observations using by .drop() in order to eliminate the last data which could have spoiled the analysis: 

# In[41]:


# Cleaning the dataset from the observations with typos
df = df.drop(df[(df['energy_rating']>=999) | (df['gas_rating']>=999) | (df['toilets']>5) | (df['terraces']>10) | (df['parking']>5)].index)

df.describe()


# ## 5. Saving the file
# Once the cleaning part over, last step is to export the cleaned dataset with .to_csv() for analysis

# In[42]:


df.to_csv('data_seloger_preparation_part2.csv',index=False)

