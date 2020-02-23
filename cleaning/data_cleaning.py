#!/usr/bin/env python
# coding: utf-8


# ### Data Cleaning ### #

# Now that the scraping part is finished, raw data extracted still need
# some cleaning and processing before being analyzed.
# Indeed as we will see below, there are some features like 'rent_info', 'criteria' or 'description'
# that holds a lot of information that needs to be extracted. We determined which information to extract
# based on some visual inspection of the listings but also based on intuition.


import matplotlib.pyplot as plt
import numpy as np
# Import libraries
import pandas as pd
import seaborn as sns
import unidecode

plt.style.use(style='ggplot')


# ## Import the data
# df = pd.read_csv(r'C:/Users/jerem/Google Drive/Mes Documents/Travail/Projects/Toulouse_Apt_Rental_Price/data/data_seloger_raw.csv')
df = pd.read_csv(
    'https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/data/data_seloger_raw.csv')


# ## Quick inspection of the features

# Size of the dataset
print("Initial data size is: {} ".format(df.shape))

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


# ## Features Extraction

# To facilitate the extraction of the information from the features 'details','rent_info','criteria','description'
# we will for each of them:
# - lower-case each string
# - remove accents from each string
# - remove '-' characters from each string

# Create a function to lower-case, remove accents and '-' characters
def text_cleaning_func(df):
    for columns in ['details', 'rent_info', 'criteria', 'description']:
        df[columns] = [unidecode.unidecode(i) for i in df[columns]]
        df[columns] = df[columns].str.replace('-', ' ')
        df[columns] = df[columns].str.lower()
    return df


# Create a function that extracts hidden features on additional rental price information
def rent_info_features_func(df):
    # Create a renting provisions amount feature
    df['provisions'] = df['rent_info'].str.extract(
        'provisions pour charges avec regularisation annuelle [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
    df['provisions'] = df['provisions'].str.replace(',', '')

    # Create a renting fees feature
    df['fees'] = df['rent_info'].str.extract(
        'honoraires ttc a la charge du locataire [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
    df['fees'] = df['fees'].str.replace(',', '')

    # Create a deposit amount feature
    df['deposit'] = df['rent_info'].str.extract('depot de garantie [:] ([\d]{0,},?[\d]{1,}.?[\d]{0,})')
    df['deposit'] = df['deposit'].str.replace(',', '')
    return df


# Create a function that returns the max between two features and drop the initial features compared
def max_func(df, feat_criteria, feat_description):
    feat = df[[feat_criteria, feat_description]].max(axis=1)
    df.drop([feat_criteria, feat_description], axis=1, inplace=True)
    return feat


# Create a function that extracts hidden features related to the inside of the apartment
def inside_features_func(df):
    ## APARTMENT
    # Create a total area size in square meter feature
    df['area'] = df['criteria'].str.extract('surface de ([\d]{0,},?[\d]{1,}) m2')
    df['area'] = df['area'].str.replace(',', '.')

    # Create a total number of rooms feature
    df['rooms'] = df['criteria'].str.extract('([\d]{1,}) piece')
    df["rooms"] = df["rooms"].fillna('0')  # if nan then 0

    # Create an entrance feature (dummy variable)
    df['entrance'] = df['criteria'].str.contains('; entree ;', regex=True).astype(int)

    # Create a duplex feature (dummy variable)
    df['duplex_crit'] = df['criteria'].str.contains('; duplex ;', regex=True).astype(int)
    df['duplex_descr'] = df['description'].str.contains('; duplex ;', regex=True).astype(int)
    df['duplex'] = max_func(df, 'duplex_crit', 'duplex_descr')

    ## LIVING ROOM
    # Create a living room feature (dummy variable)
    df['livingroom'] = df['criteria'].str.contains('; sejour ;', regex=True).astype(int)

    # Create a total living room area size in square meter feature
    df['livingroom_area'] = df['criteria'].str.extract('sejour de ([\d]{0,},?[\d]{1,}) m2')
    df["livingroom_area"] = df["livingroom_area"].fillna('0')  # if nan then 0

    ## KITCHEN
    # Create an equipped kitchen feature (dummy variable)
    df['equipped_kitchen_crit'] = df['criteria'].str.contains('cuisine equipe|cuisine americaine equipe',
                                                              regex=True).astype(int)
    df['equipped_kitchen_descr'] = df['description'].str.contains('cuisine equipe|cuisine américaine equipe',
                                                                  regex=True).astype(int)
    df['equipped_kitchen'] = max_func(df, 'equipped_kitchen_crit', 'equipped_kitchen_descr')

    # Create an open-plan kitchen feature (dummy variable)
    df['openplan_kitchen_crit'] = df['criteria'].str.contains('cuisine americaine', regex=True).astype(int)
    df['openplan_kitchen_descr'] = df['description'].str.contains('cuisine americaine', regex=True).astype(int)
    df['openplan_kitchen'] = max_func(df, 'openplan_kitchen_crit', 'openplan_kitchen_descr')

    ## BEDROOMS
    # Create a total number of bedrooms feature
    df['bedrooms'] = df['criteria'].str.extract('([\d]{1,}) chambre')
    df["bedrooms"] = df["bedrooms"].fillna('0')  # if nan then 0

    ## BATHROOMS
    # Create a total number of bathrooms feature
    df['bathrooms'] = df['criteria'].str.extract('([\d]{1,}) salle de bain')
    df["bathrooms"] = df["bathrooms"].fillna('0')  # if nan then 0

    # Create a total number of shower rooms feature
    df['shower_rooms'] = df['criteria'].str.extract('([\d]{1,}) salle d\'eau')
    df["shower_rooms"] = df["shower_rooms"].fillna('0')  # if nan then 0

    ## TOILETS
    # Create a total number of toilets feature
    df['toilets'] = df['criteria'].str.extract('([\d]{1,}) toilette')

    # Create a separate toilet feature (dummy variable)
    df['separate_toilet_crit'] = df['criteria'].str.contains('toilettes separe', regex=True).astype(int)
    df['separate_toilet_descr'] = df['description'].str.contains('toilettes separe', regex=True).astype(int)
    df['separate_toilet'] = max_func(df, 'separate_toilet_crit', 'separate_toilet_descr')

    ## BALCONY, TERRACES
    # Create a total number of balconies feature
    df['balcony'] = df['criteria'].str.extract('([\d]{1,}) balcon')
    df["balcony"] = df["balcony"].fillna('0')  # if nan then 0

    # Create a total number of terraces feature
    df['terraces'] = df['criteria'].str.extract('([\d]{1,}) terrasse')
    df["terraces"] = df["terraces"].fillna('0')  # if nan then 0

    ## WOODEN FLOOR, FIREPLACE, INSIDE STORAGE
    # Create a wooden floor feature (dummy variable)
    df['wooden_floor'] = df['criteria'].str.contains('parquet', regex=True).astype(int)

    # Create a fireplace feature (dummy variable)
    df['fireplace_crit'] = df['criteria'].str.contains('cheminee', regex=True).astype(int)
    df['fireplace_descr'] = df['description'].str.contains('cheminee', regex=True).astype(int)
    df['fireplace'] = max_func(df, 'fireplace_crit', 'fireplace_descr')

    # Create an inside storage feature (dummy variable)
    df['storage'] = df['criteria'].str.contains('rangement', regex=True).astype(int)

    ## HEATING
    # Create an heating function
    def heating_func(df):
        df['heating'] = np.nan
        if 'chauffage gaz collectif' in df['criteria'] or 'chauffage au gaz collectif' in df[
            'criteria'] or 'chauffage gaz collectif' in df['description'] or 'chauffage au gaz collectif' in df[
            'description']:
            return 'collective_gas'
        elif 'chauffage individuel gaz' in df['criteria'] or 'chauffage individuel au gaz' in df[
            'criteria'] or 'chauffage individuel gaz' in df['description'] or 'chauffage individuel au gaz' in df[
            'description']:
            return 'individual_gas'
        elif 'chauffage gaz' in df['criteria'] or 'chauffage au gaz' in df['criteria'] or 'chauffage gaz' in df[
            'description'] or 'chauffage au gaz' in df['description']:
            return 'gas'
        elif 'chauffage individuel electrique' in df['criteria'] or 'chauffage electrique' in df[
            'criteria'] or 'chauffage individuel electrique' in df['description'] or 'chauffage electrique' in df[
            'description']:
            return 'individual_electrical'
        elif 'chauffage individuel' in df['criteria'] or 'chauffage individuel' in df['description']:
            return 'individual'
        elif 'chauffage central' in df['criteria'] or 'chauffage central' in df['description']:
            return 'central'
        elif 'chauffage sol' in df['criteria'] or 'chauffage au sol' in df['criteria'] or 'chauffage sol' in df[
            'description'] or 'chauffage au sol' in df['description']:
            return 'central'
        else:
            return df['heating']

    df['heating'] = df.apply(heating_func, axis=1)
    return df


# Create a function that extracts hidden features related to the outside of the apartment
def outside_features_func(df):
    # Create a parking feature
    df['parking'] = df['criteria'].str.extract('([\d]{1,}) parking')
    df["parking"] = df["parking"].fillna('0')  # if nan then 0

    # Create a cellar feature (dummy variable)
    df['cellar'] = df['criteria'].str.contains('; cave ;', regex=True).astype(int)

    # Create a pool feature (dummy variable)
    df['pool_crit'] = df['criteria'].str.contains('piscine', regex=True).astype(int)
    df['pool_descr'] = df['description'].str.contains('piscine', regex=True).astype(int)
    df['pool'] = max_func(df, 'pool_crit', 'pool_descr')

    # Create a year of construction of the building feature
    df['construction_year'] = df['criteria'].str.extract('annee de construction ([\d]{4,4})')

    # Create the total number of floors of the building feature
    df['bldg_flr_nb'] = df['criteria'].str.extract('batiment de ([\d]{1,}) etage')

    # Create the floor number of the appartment feature
    df['apt_flr_nb'] = df['criteria'].str.extract('; au ([\d]{1,})+(?:er|eme) etage')
    df.loc[df['criteria'].str.contains('au rez-de-chausse', case=False), 'apt_flr_nb'] = '0'
    return df


# Create a function that extracts hidden features related to the advantages of the apartment
def advantages_features_func(df):
    ## FURNISHED, RENOVATED
    # Create a furnished feature (dummy variable)
    df['furnished'] = df['criteria'].str.contains('; meuble ;', regex=True).astype(int)

    # Create a renovated feature (dummy variable)
    df['renovated'] = df['criteria'].str.contains('refait a neuf', regex=True).astype(int)

    ## ELEVATOR, INTERCOM, DIGITAL CODE
    # Create an elevator feature (dummy variable)
    df['elevator'] = df['criteria'].str.contains('ascenseur', regex=True).astype(int)

    # Create an intercom feature (dummy variable)
    df['intercom'] = df['criteria'].str.contains('interphone', regex=True).astype(int)

    # Create a digital code feature (dummy variable)
    df['digital_code'] = df['criteria'].str.contains('digicode', regex=True).astype(int)

    ## ORIENTATION, VIEW
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

    ## CARETAKER, REDUCED-MOBILITY PERSONS
    # Create a caretaker feature (dummy variable)
    df['caretaker'] = df['criteria'].str.contains('; gardien ;', regex=True).astype(int)

    # Create a reduced mobility feature (dummy variable)
    df['reduced_mobility'] = df['criteria'].str.contains('adapte pmr', regex=True).astype(int)
    return df


#  Neighborhood information
# Due to the lack of consistency of the descriptions we need to use
# both websites https://www.toulouse.fr/vos-quartiers and https://fr.wikipedia.org/wiki/Quartiers_de_Toulouse
# to help us create the right neighborhoods and sectors.
# We learn from those websites that Toulouse is divided in 6 different sectors
# and each one of them is also divided in 3 to 4 neighborhoods for a total of 20 different neighborhoods.

# Create a dictionary to change the neighborhood codes to neighborhood names
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

# Create a dictionary to change the neighborhood codes to sector codes
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

# Create a dictionary to change the sector codes to sector names
sector_dict = {'sector1': 'TOULOUSE CENTRE',
               'sector2': 'TOULOUSE RIVE GAUCHE',
               'sector3': 'TOULOUSE NORD',
               'sector4': 'TOULOUSE EST',
               'sector5': 'TOULOUSE SUD EST',
               'sector6': 'TOULOUSE OUEST'}

# Below we assign for each neighborhood some keywords
# that we will look for in both features 'description' and 'details'
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


# Create a function that extracts the neighborhood name and code
def neighborhood_features_func(df):
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

    # Create a neighborhood name from the dictionary neighborhood_dict
    df['nbhd_name'] = df['nbhd_no'].map(neighborhood_dict)
    return df


# Create a function that extracts the sector name and code based on the neighborhood name
def sector_features_func(df):
    # Create a sector code from the dictionary neighborhood_dict
    df['sector_no'] = df['nbhd_no'].map(neighborhood_sector_dict)

    # Create a sector name from the dictionary neighborhood_dict
    df['sector_name'] = df['sector_no'].map(sector_dict)
    return df


# Create a function that extracts the postal code
def postal_features_func(df):
    # Create a postal code feature
    df['postal_code'] = df['description'].str.extract('(31[\d]{3,3})')
    return df


# Create a function that extracts hidden features related to public transportation around the apartment
def transportation_features_func(df):
    # Create a metro feature (dummy variable)
    df['metro'] = df['description'].str.contains('metro', regex=True).astype(int)

    # Create a metro feature (dummy variable)
    df['tram'] = df['description'].str.contains('tram|tramway', regex=True).astype(int)

    # Create a metro feature (dummy variable)
    df['bus'] = df['description'].str.contains('bus|bus.|bus,', regex=True).astype(int)
    return df


# ## Columns Transformation

# Create a function that reorder columns, drop useless ones and convert to numeric columns that can be converted
def clean_columns_func(df):
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

    # Convert to numeric columns that can be converted using apply method
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
    return df


# ## Remove Duplicates

# As discussed in the previous part we found some duplicate entries based on the column 'link'
# We need to delete duplicate entries in the dataset as they would affect our analysis
# as our learning algorithm would learn from incorrect data.

# Create a function that removes duplicate values
def deduplicate_func(df):
    # Finding out duplicates
    uniqueRows = len(set(df.link))
    totalRows = len(df.link)
    duplicateRows = totalRows - uniqueRows
    print("Data size before dropping duplicate values is: {} ".format(df.shape))
    print('There are {} duplicates'.format(duplicateRows))

    # dropping duplicate values
    df = df.drop_duplicates(subset='link', keep="first")

    print("Data size after dropping duplicate values is: {} ".format(df.shape))
    return df


# ## Remove Outliers

# Outliers will sit way outside of the distribution of data points
# and skew the distribution of the data and potential calculations.
# Therefore we need to identify and remove them

# Create a function that plot and remove outliers
def remove_outliers_func(df):
    # Explore outliers
    fig, ax = plt.subplots()
    ax.scatter(df['area'], df['rent'], color='blue')
    plt.ylabel('rent', fontsize=13)
    plt.xlabel('area', fontsize=13)
    plt.show()

    # Cleaning the dataset from its outliers
    df = df.drop(df[(df['area'] < 40) & (df['rent'] > 1500)].index)

    # Check data after removing outliers
    fig, ax = plt.subplots()
    ax.scatter(df['area'], df['rent'], color='blue')
    plt.ylabel('rent', fontsize=13)
    plt.xlabel('area', fontsize=13)
    plt.show()
    return df


# ## Filling Missing Values

# Missing values are the Data Scientists other nightmare. They can mean multiple things:
# - A missing value may be the result of an error during the production of the dataset.
#   Depending on where the data comes from, this could be:
#     - a human error
#     - a machinery error
# - A missing value in some cases, may just mean a that a 'zero' should be present.
#   In which case, it can be replaced by a 0.
#   The data description provided helps to address situations like these.
# - Otherwise, missing values represent no information.
#   Therefore, does the fact that you don't know what value to assign an entry,
#   mean that filling it with a 'zero' is always a good fit?
#
# Some algorithms do not like missing values. Some are capable of handling them, but others are not.
# Therefore since we are using a variety of algorithms, it's best to treat them in an appropriate way.
# If you have missing values, you have two options:
# - Delete the entire row
# - Fill the missing entry with an imputed value

# In order to treat this dataset we will cycle through each feature with missing values
# and treat them individually based on the data description, or our judgement.
# Through reference of the data description, this gives guidance on how to treat missing values for some columns.
# For ones where guidance isn't clear enough, we have to use intuition.

# Create a function that plot and treat missing values accordingly to each feature characteristics
def fill_missing_values_func(df):
    # Inspection the missing values
    df_na = (df.isnull().sum() / len(df)) * 100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=df_na.index, y=df_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()

    # Table of the missing values
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_dataSubset = missing_data[missing_data['Total'] > 0]
    print(missing_dataSubset)

    # Drop columns with more than 40% of missing values
    df.drop(['postal_code', 'orientation', 'bldg_flr_nb', 'construction_year', 'provisions'], axis=1, inplace=True)

    # - 'nbhd_no' / 'nbhd_name' : According to data description missing values are due to a lack of information
    # Replacing missing data with the mode of 'neighborhood'
    df['nbhd_no'] = df['nbhd_no'].fillna(df['nbhd_no'].mode()[0])
    df['nbhd_name'] = df['nbhd_no'].map(neighborhood_dict)

    # - 'sector_no' / 'sector_name' : According to data description missing values are due to a lack of information
    # Replacing missing data with the appropriate sector according to the neighborhood the apartment is in.
    df = sector_features_func(df)

    # - 'agency': According to data description missing values are due to the the fact that
    # some apartment are not rented via a real estate agency
    # Replacing missing data with 'None'
    df['agency'] = df['agency'].fillna('None')

    # - 'fees' : According to data description missing values are, if rented via a real estate agency,
    # due to a lack of information and since fees of each apartment most likely have similar fees to other apartments
    # in its neighborhood we can fill in missing values by the median of the neighborhood if no agency, else replace with 0
    df.loc[(df['fees'].isnull()) & (df.agency == 'None'), 'fees'] = 0
    df['fees'] = df.groupby('nbhd_no')['fees'].transform(lambda x: x.fillna(x.median()))

    # - 'deposit', 'energy_rating', 'gas_rating', 'apt_flr_nb', 'toilets', 'area': According to data description
    # missing values are due to a lack of information and since the value of this feature for each apartment
    # most likely have a similar value than other apartments in its neighborhood
    # we can fill in missing values by the median of the neighborhood
    for col in ('deposit', 'energy_rating', 'gas_rating', 'apt_flr_nb', 'toilets', 'area'):
        df[col] = df.groupby('nbhd_no')[col].transform(lambda x: x.fillna(x.median()))

    # - 'heating': According to data description missing values are due to a lack of information.
    # We can fill in missing values by the mode 'heating' of the neighborhood
    df['heating'] = df.groupby('nbhd_no')['heating'].transform(lambda x: x.fillna(x.mode()[0]))

    # Final inspection of missing values
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm')
    plt.show()

    print('There are {} missing value(s) left'.format(sum(df.isnull().sum())))
    return df


# ## Remove human mistakes

# Create a function that cleans data from inconsistent values
def remove_mistakes_func(df):
    df.describe().head()
    # We deduce here that the work is not finished yet. Indeed we observe the presence of some issues:
    # - the highest energy rating indicated is above the legal maximum.
    # - the highest gas rating indicated is above the legal maximum.
    # - the maximum number of toilets is 10.
    # - the maximum number of terraces places is 21.
    # - the maximum number of parking places is 29.
    # Drop those observations presenting human mistakes
    df = df.drop(df[(df['energy_rating'] >= 999) | (df['gas_rating'] >= 999) | (df['toilets'] > 5) | (
            df['terraces'] > 10) | (df['parking'] > 5)].index)

    df.describe().head()
    return df


# ## Return Clean Data

# Create a final function that returns a clean dataframe based on functions created previously
def data_clean_func(df):
    print("Data size before data preparation is: {} ".format(df.shape))

    df = text_cleaning_func(df)
    df = rent_info_features_func(df)
    df = inside_features_func(df)
    df = outside_features_func(df)
    df = advantages_features_func(df)
    df = neighborhood_features_func(df)
    df = sector_features_func(df)
    df = postal_features_func(df)
    df = transportation_features_func(df)
    df = clean_columns_func(df)
    df = deduplicate_func(df)
    df = remove_outliers_func(df)
    df = fill_missing_values_func(df)
    df = remove_mistakes_func(df)

    print("Data size after data preparation is: {} ".format(df.shape))
    return df


df = data_clean_func(df)


### Export the file
df.to_csv('data_seloger_clean.csv', index=False)
print("Data exported")
