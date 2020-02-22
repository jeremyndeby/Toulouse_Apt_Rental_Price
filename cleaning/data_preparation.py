#!/usr/bin/env python
# coding: utf-8


##### Data Cleaning #####

# Now that the scraping part is finished, raw data extracted still need
# some cleaning and processing before being analyzed.

# Indeed as we will see below, there are some features like 'rent_info', 'criteria' or 'description'
# that holds a lot of information that needs to be extracted. # We determined which information to extract
# based on some visual inspection of the listings but also based on experience.


import numpy as np
# Import libraries
import pandas as pd
import unidecode

# from IPython.display import display
# import seaborn

### Import the data
# df = pd.read_csv(r'C:/Users/jerem/Google Drive/Mes Documents/Travail/Projects/Toulouse_Apt_Rental_Price/data/data_seloger_raw.csv')
df = pd.read_csv(
    'https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/data/data_seloger_raw.csv')

### Quick inspection of the features

# Size of the dataset
df.shape

#  Columns in the dataset
print(df.columns)

#  columns in the dataset:
print(df.columns)

# Inspect the link feature
print(df['link'].describe())
# Here we can see that some listings were published several times (up to four times)
# over the different listing pages we extracted the data from.
# In the next part we will need to deduplicate our dataset based on this feature.

# Inspect the title feature
print(df['title'].describe())
for i in range(5):
    print(df.loc[i, 'title'])
# From the 'title' feature we can extract:
# - the number of rooms of the apartment
# - the city
# - the size of the apartment in square meters
# - the price of the rent

# Inspect the agency feature
print(df['agency'].describe())
print(df['agency'].value_counts())
# There are 142 different real estate agencies in our dataset with SNG Extensia the agency
# with the highest number of apartment listed (before deduplication).
# There seems also to be a share of the apartment without an agency
# meaning that the landlord does not go through an agency to rent its apartment.
# This feature might play a double role on the rent as: 
# - going through a real estate agenxy or not might have an impact
# on the rent as agencies take a commission on each appartment rented
# - the size and reputation of the agency might have an impact
# on the rent as renowned agencies might take an higher comission on each appartment rented

# Inspect the city feature
print(df['city'].describe())
print(df['city'].value_counts())
# We can see four different cities displayed but when taking a closer look
# at each unique value we can deduce there are all refering to Toulouse:
# - 'Toulouse.': typo, same as Toulouse
# - '31000': One of Toulouse's postal code
# - 'Saint Martin du Touch': former autonomous village that has become a neighborhood of Toulouse.
# It is part of sub-division 6.1 of sector 6 of the city.
# This is not surprising as our goal was to extract data from apartment in Toulouse only.
# This feature will then probably be dropped later.

# Inspect the housing type feature
print(df['housing_type'].describe())
# There is only one housing type in the dataset.
# This is not surprising as our goal was to extract data from apartment listings only.
# This feature will probably be dropped later.

# Inspect the details feature
print(df['details'].describe())
for i in range(5):
    print(df.loc[i, 'details'])
# From the 'details' feature we can extract:
# - the size of the apartment in square meters
# - the neighborhood/sector of the apartment

# Inspect the rent feature
print(df['rent'].describe())
# The rent feature does not need any transformation. 

# Inspect the charges feature
print(df['charges'].value_counts())
# Only two listings does not include charges in the rent (before deduplication).
# This feature will probably be dropped later.

# Inspect the rent information feature
print(df['rent_info'].describe())
for i in range(3):
    print(df.loc[i, 'rent_info'])
# Most of the listings do not have any rent information but when available we can extract: 
# - The renting provisions amount for the apartment
# - The renting fees amount for the apartment
# - The deposit amount for the apartment

# Inspect the criteria feature
print(df['criteria'].describe())
for i in range(3):
    print(df.loc[i, 'criteria'])
# From this feature we will extract quite a few information such as some:
# - Inside features
# - Outside features
# - Advantages 
# - Neighborhood/Sector features

# Inspect the energy rating feature
print(df['energy_rating'].describe())
# The energy rating feature does not need any transformation.

# Inspect the gas rating feature
print(df['gas_rating'].describe())
# The gas rating feature does not need any transformation.

# Inspect the description feature
print(df['description'].describe())
for i in range(5):
    print(df.loc[i, 'description'])


# From this feature we will also extract quite a few information from different type such as:
# - Inside features
# - Outside features
# - Advantages 
# - Neighborhood/Sector features


### Data Transformation

# To facilitate the extraction of the information from the features 'details','rent_info','criteria','description'
# we will for each of them:
# - lower-case each string
# - remove accents from each string
# - remove '-' characters from each string

# Create a function to lower-case, remove accents and '-' characters
def text_cleaning_func(data):
    for columns in ['details', 'rent_info', 'criteria', 'description']:
        data[columns] = [unidecode.unidecode(i) for i in data[columns]]
        data[columns] = data[columns].str.replace('-', ' ')
        data[columns] = data[columns].str.lower()
    return data


df = text_cleaning_func(df)


# Create a function to extract hidden features on additional rental price information
def rent_info_features_func(data):
    # Create a renting provisions amount feature
    data['provisions'] = data['rent_info'].str.extract(
        'provisions pour charges avec regularisation annuelle [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
    data['provisions'] = data['provisions'].str.replace(',', '')

    # Create a renting fees feature
    data['fees'] = data['rent_info'].str.extract(
        'honoraires ttc a la charge du locataire [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
    data['fees'] = data['fees'].str.replace(',', '')

    # Create a deposit amount feature
    data['deposit'] = data['rent_info'].str.extract('depot de garantie [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
    data['deposit'] = data['deposit'].str.replace(',', '')
    return data


df = rent_info_features_func(df)


# Create a function that returns the max between two features and drop the initial features compared
def max_func(df, feat_criteria, feat_description):
    feat = df[[feat_criteria, feat_description]].max(axis=1)
    df.drop([feat_criteria, feat_description], axis=1, inplace=True)
    return feat


# Create a function to extract hidden features based on the inside of the apartment
def inside_features_func(data):
    ## APARTMENT
    # Create a total area size in square meter feature
    data['area'] = data['criteria'].str.extract('surface de ([\d]{0,},?[\d]{1,}) m2')
    data['area'] = data['area'].str.replace(',', '.')

    # Create a total number of rooms feature
    data['rooms'] = data['criteria'].str.extract('([\d]{1,}) piece')
    data["rooms"] = data["rooms"].fillna('0')  # if nan then 0

    # Create an entrance feature (dummy variable)
    data['entrance'] = data['criteria'].str.contains('; entree ;', regex=True).astype(int)

    # Create a duplex feature (dummy variable)
    data['duplex_crit'] = data['criteria'].str.contains('; duplex ;', regex=True).astype(int)
    data['duplex_descr'] = data['description'].str.contains('; duplex ;', regex=True).astype(int)
    data['duplex'] = max_func(data, 'duplex_crit', 'duplex_descr')

    ## LIVING ROOM
    # Create a living room feature (dummy variable)
    data['livingroom'] = data['criteria'].str.contains('; sejour ;', regex=True).astype(int)

    # Create a total living room area size in square meter feature
    data['livingroom_area'] = data['criteria'].str.extract('sejour de ([\d]{0,},?[\d]{1,}) m2')
    data["livingroom_area"] = data["livingroom_area"].fillna('0')  # if nan then 0

    ## KITCHEN
    # Create an equipped kitchen feature (dummy variable)
    data['equipped_kitchen_crit'] = data['criteria'].str.contains('cuisine equipe|cuisine americaine equipe',
                                                                  regex=True).astype(int)
    data['equipped_kitchen_descr'] = data['description'].str.contains('cuisine equipe|cuisine américaine equipe',
                                                                      regex=True).astype(int)
    data['equipped_kitchen'] = max_func(data, 'equipped_kitchen_crit', 'equipped_kitchen_descr')

    # Create an open-plan kitchen feature (dummy variable)
    data['openplan_kitchen_crit'] = data['criteria'].str.contains('cuisine americaine', regex=True).astype(int)
    data['openplan_kitchen_descr'] = data['description'].str.contains('cuisine americaine', regex=True).astype(int)
    data['openplan_kitchen'] = max_func(data, 'openplan_kitchen_crit', 'openplan_kitchen_descr')

    ## BEDROOMS
    # Create a total number of bedrooms feature
    data['bedrooms'] = data['criteria'].str.extract('([\d]{1,}) chambre')
    data["bedrooms"] = data["bedrooms"].fillna('0')  # if nan then 0

    ## BATHROOMS
    # Create a total number of bathrooms feature
    data['bathrooms'] = data['criteria'].str.extract('([\d]{1,}) salle de bain')
    data["bathrooms"] = data["bathrooms"].fillna('0')  # if nan then 0

    # Create a total number of shower rooms feature
    data['shower_rooms'] = data['criteria'].str.extract('([\d]{1,}) salle d\'eau')
    data["shower_rooms"] = data["shower_rooms"].fillna('0')  # if nan then 0

    ## TOILETS
    # Create a total number of toilets feature
    data['toilets'] = data['criteria'].str.extract('([\d]{1,}) toilette')

    # Create a separate toilet feature (dummy variable)
    data['separate_toilet_crit'] = data['criteria'].str.contains('toilettes separe', regex=True).astype(int)
    data['separate_toilet_descr'] = data['description'].str.contains('toilettes separe', regex=True).astype(int)
    data['separate_toilet'] = max_func(data, 'separate_toilet_crit', 'separate_toilet_descr')

    ## BALCONY, TERRACES
    # Create a total number of balconies feature
    data['balcony'] = data['criteria'].str.extract('([\d]{1,}) balcon')
    data["balcony"] = data["balcony"].fillna('0')  # if nan then 0

    # Create a total number of terraces feature
    data['terraces'] = data['criteria'].str.extract('([\d]{1,}) terrasse')
    data["terraces"] = data["terraces"].fillna('0')  # if nan then 0

    ## WOODEN FLOOR, FIREPLACE, INSIDE STORAGE
    # Create a wooden floor feature (dummy variable)
    data['wooden_floor'] = data['criteria'].str.contains('parquet', regex=True).astype(int)

    # Create a fireplace feature (dummy variable)
    data['fireplace_crit'] = data['criteria'].str.contains('cheminee', regex=True).astype(int)
    data['fireplace_descr'] = data['description'].str.contains('cheminee', regex=True).astype(int)
    data['fireplace'] = max_func(data, 'fireplace_crit', 'fireplace_descr')

    # Create an inside storage feature (dummy variable)
    data['storage'] = data['criteria'].str.contains('rangement', regex=True).astype(int)

    ## HEATING
    # Create an heating function
    def heating_func(data):
        data['heating'] = np.nan
        if 'chauffage gaz collectif' in data['criteria'] or 'chauffage au gaz collectif' in data[
            'criteria'] or 'chauffage gaz collectif' in data['description'] or 'chauffage au gaz collectif' in data[
            'description']:
            return 'collective_gas'
        elif 'chauffage individuel gaz' in data['criteria'] or 'chauffage individuel au gaz' in data[
            'criteria'] or 'chauffage individuel gaz' in data['description'] or 'chauffage individuel au gaz' in data[
            'description']:
            return 'individual_gas'
        elif 'chauffage gaz' in data['criteria'] or 'chauffage au gaz' in data['criteria'] or 'chauffage gaz' in data[
            'description'] or 'chauffage au gaz' in data['description']:
            return 'gas'
        elif 'chauffage individuel electrique' in data['criteria'] or 'chauffage electrique' in data[
            'criteria'] or 'chauffage individuel electrique' in data['description'] or 'chauffage electrique' in data[
            'description']:
            return 'individual_electrical'
        elif 'chauffage individuel' in data['criteria'] or 'chauffage individuel' in data['description']:
            return 'individual'
        elif 'chauffage central' in data['criteria'] or 'chauffage central' in data['description']:
            return 'central'
        elif 'chauffage sol' in data['criteria'] or 'chauffage au sol' in data['criteria'] or 'chauffage sol' in data[
            'description'] or 'chauffage au sol' in data['description']:
            return 'central'
        else:
            return data['heating']

    data['heating'] = data.apply(heating_func, axis=1)
    return data


df = inside_features_func(df)


# Create a function to extract hidden features based on the outside of the apartment
def outside_features_func(data):
    # Create a parking feature
    data['parking'] = data['criteria'].str.extract('([\d]{1,}) parking')
    data["parking"] = data["parking"].fillna('0')  # if nan then 0

    # Create a cellar feature (dummy variable)
    data['cellar'] = data['criteria'].str.contains('; cave ;', regex=True).astype(int)

    # Create a pool feature (dummy variable)
    data['pool_crit'] = data['criteria'].str.contains('piscine', regex=True).astype(int)
    data['pool_descr'] = data['description'].str.contains('piscine', regex=True).astype(int)
    data['pool'] = max_func(data, 'pool_crit', 'pool_descr')

    # Create a year of construction of the building feature
    data['construction_year'] = data['criteria'].str.extract('annee de construction ([\d]{4,4})')

    # Create the total number of floors of the building feature
    data['bldg_flr_nb'] = data['criteria'].str.extract('batiment de ([\d]{1,}) etage')

    # Create the floor number of the appartment feature
    data['apt_flr_nb'] = data['criteria'].str.extract('; au ([\d]{1,})+(?:er|eme) etage')
    data.loc[data['criteria'].str.contains('au rez-de-chausse', case=False), 'apt_flr_nb'] = '0'
    return data


df = outside_features_func(df)


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
def orientation_func(df):
    df['orientation'] = np.nan
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
    elif 'orientation ouest' in df['criteria']:
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
neighborhood_sector_dict = {'n1_1': 'sector1',
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
sector_dict = {'sector1': 'TOULOUSE CENTRE',
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
# 1.1 : Capitole - Arnaud Bernard - Carmes
# 1.2 : Amidonniers - Compans Caffarelli
# 1.3 : Les Chalets/Bayard/Belfort  Saint-Aubin/Dupuy

n1_1 = ['capitole', 'bernard', 'carmes',
        'jacobins', ' rome', 'dalbade', 'filatiers', 'sernin', 'taur',
        'valade', 'occitane', 'wilson', 'baragnon', 'ozenne', 'trinite', 'esquirol',
        'georges', 'fac de droit', 'jaures', 'jeanne d\'arc',
        'st pierre', 'saint pierre', 'francois verdier', 'la daurade', 'lascrosses',
        'saint etienne', 'st etienne', 'alsace lorraine', 'pont neuf', 'alsace', 'lorraine',
        'place de la bourse', 'toulouse centre', 'hyper centre', 'hypercentre']

n1_2 = ['amidonniers', 'compans', 'caffarelli',
        'bazacle', 'chapou', 'heracles', 'sébastopol', 'barcelone', 'brienne']

n1_3 = ['chalets', 'bayard', 'belfort', 'aubin', 'dupuy',
        'concorde', 'raymond iv', 'belfort', 'gabriel peri', 'colombette', 'grand rond',
        'honore serres', 'matabiau']

## Secteur 2: TOULOUSE RIVE GAUCHE
# 2.1 : Saint-Cyprien
# 2.2 : Croix de Pierre - Route d'Espagne
# 2.3 : Fontaine-Lestang - Arènes -Bagatelle - Papus - Tabar - Bordelongue - Mermoz - La Faourette
# 2.4 : Casselardit - Fontaine-Bayonne - Cartoucherie

n2_1 = ['cyprien',
        'bourrassol', 'la grave', 'ravelin', 'roguet', ' lucie', 'teinturiers', 'patte d\'oie', 'patte oie',
        'fer a cheval', 'abattoirs', 'toulouse rive gauche']

n2_2 = ['croix de pierre', 'route d\'espagne',
        'becanne', 'la digue', 'la pointe', 'avenue de muret', 'oustalous', 'deodat de severac']

n2_3 = ['lestang', 'arenes', 'bagatelle', ' papus', 'tabar', 'bordelongue', 'mermoz', 'farouette',
        'arenes', 'bigorre', 'lambert', ' loire', 'morvan', ' tellier', ' touraine', 'vestrepain', 'hippodrome']

n2_4 = ['casselardit', 'fontaine-bayonne', 'fontaine bayonne', 'cartoucherie',
        'barrière de bayonne', 'biarritz', 'les fontaines', 'zenith']

## Secteur 3: TOULOUSE NORD
# 3.1 : Minimes - Barrière de Paris - Ponts-Jumeaux
# 3.2 : Sept Deniers - Ginestous - Lalande
# 3.3 : Trois Cocus - Borderouge - Croix Daurade - Paleficat - Grand Selve

n3_1 = ['minimes', 'barriere de paris', 'jumeaux',
        'la vache', 'canal du midi']

n3_2 = ['deniers', 'ginestous', 'lalande',
        'route de launaguet']

n3_3 = ['trois cocus', '3 cocus', 'borderouge', 'croix daurade', 'paleficat', 'grand selve',
        'izards']

## Secteur 4: TOULOUSE EST
# 4.1 : Lapujade - Bonnefoy - Périole - Marengo - La Colonne
# 4.2 : Jolimont - Soupetard - Roseraie - Gloire - Gramont- Amouroux
# 4.3 : Bonhoure - Guilheméry - Château de l'Hers - Limayrac - Côte Pavée

n4_1 = ['lapujade', 'bonnefoy', 'periole', 'marengo', 'la colonne']

n4_2 = ['jolimont', 'soupetard', 'roseraie', 'gloire', 'gramont', 'amouroux',
        'argoulets']

n4_3 = ['bonhoure', 'guilhemery', 'l\'hers', 'limayrac', 'cote pavee',
        'camille pujol', 'hers', 'grande plaine']

## Secteur 5: TOULOUSE SUD EST
# 5.1 : Pont des Demoiselles - Ormeau - Montaudran - La Terrasse - Malepère
# 5.2 : Rangueil - Saouzelong - Pech David - Pouvourville
# 5.3 :Saint Michel - Le Busca - Empalot - Saint Agne

n5_1 = ['demoiselles', 'ormeau', 'montaudran', 'la terrasse ', 'malepere',
        'exupery']

n5_2 = ['rangueil', 'saouzelong', 'pech david', 'pouvourville',
        'faculte de pharmacie', 'paul sabatier', 'ramonville', 'rangeuil', 'palays', 'jules julien']

n5_3 = ['michel', 'busca', 'empalot', ' agne',
        'palais de justice', 'des plantes', 'ramier']

## Secteur 6: TOULOUSE OUEST
# 6.1 : Arènes Romaines - Saint Martin du Touch- Purpan
# 6.2 : Lardenne - Pradettes - Basso Cambo
# 6.3 : Mirail- Université - Reynerie - Bellefontaine
# 6.4 : Saint Simon - Lafourguette - Oncopole


n6_1 = ['arenes romaines', 'martin ', 'du touch', 'purpan']

n6_2 = ['lardenne', 'pradettes', 'basso',
        'bordeblanche', 'cepiere']

n6_3 = ['mirail', 'reynerie', 'bellefontaine']

n6_4 = ['simon', 'lafourguette', 'oncopole',
        'ramee']


# Let's now create a function to search for the keywords we identified above and return the neighborhood code:

# In[34]:


# Create a neighborhood function for the 'description' feature
def neighborhood_description_func(df):
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
def neighborhood_details_func(df):
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
df.drop(['nbhd_no_description', 'nbhd_no_details'], axis=1, inplace=True)

# Now we use the dictionaries create before to create a neighborhood name, sector code and sector name from the neighborhood code:

# In[35]:


# Create a neighborhood name from the dictionary neighborhood_dict
df['nbhd_name'] = df['nbhd_no'].map(neighborhood_dict)

# Create a sector code from the dictionary neighborhood_dict
df['sector_no'] = df['nbhd_no'].map(neighborhood_sector_dict)

# Create a sector name from the dictionary neighborhood_dict
df['sector_name'] = df['sector_no'].map(sector_dict)

# Check the frequency
# df.groupby(["sector_no", "nbhd_no"]).size()


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
df.drop(['title', 'housing_type', 'city', 'details', 'rent_info', 'criteria', 'description'], axis=1, inplace=True)

# Reorder columns
df = df[['link', 'agency', 'postal_code', 'sector_no', 'sector_name', 'nbhd_no', 'nbhd_name',
         'rent', 'charges', 'provisions', 'fees', 'deposit',
         'energy_rating', 'gas_rating',
         'area', 'rooms', 'entrance', 'duplex',
         'livingroom', 'livingroom_area', 'equipped_kitchen', 'openplan_kitchen',
         'bedrooms', 'bathrooms', 'shower_rooms', 'toilets', 'separate_toilet',
         'balcony', 'terraces', 'wooden_floor', 'fireplace', 'storage', 'heating',
         'parking', 'cellar', 'pool',
         'construction_year', 'bldg_flr_nb', 'apt_flr_nb',
         'furnished', 'renovated', 'elevator', 'intercom', 'digital_code', 'orientation', 'view', 'caretaker',
         'reduced_mobility',
         'metro', 'tram', 'bus']]

# We also check data type with .info()

# In[38]:


df.info()

# We convert to numeric columns that can be converted:

# In[39]:


# using apply method 
df[['rent', 'provisions', 'fees', 'deposit',
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
                                   'furnished', 'renovated', 'elevator', 'intercom', 'digital_code', 'view',
                                   'caretaker', 'reduced_mobility',
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
df = df.drop(df[(df['energy_rating'] >= 999) | (df['gas_rating'] >= 999) | (df['toilets'] > 5) | (
        df['terraces'] > 10) | (df['parking'] > 5)].index)

df.describe()

# ## 5. Saving the file
# Once the cleaning part over, last step is to export the cleaned dataset with .to_csv() for analysis

# In[42]:


df.to_csv('data_seloger_preparation_part2_test.csv', index=False)
