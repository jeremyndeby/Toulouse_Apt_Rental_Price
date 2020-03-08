#!/usr/bin/env python
# coding: utf-8

# ### Explanatory Data Analysis & Feature Engineering ### #

# One of the most important steps in any predictive modeling problem is exploratory data analysis.
# Exploratory data analysis is a process of using basic summary statistics and visualizations
# to gain intuition about the features of the data.
# This should always be done before applying predictive modeling algorithms to your data.

# To guide our feature engineering, we would like to know what factors add value to an apartment?
# From a quick online research, there are several key things that stood out:
# - Location - location is key for high valuations, therefore having a safe,
#   well facilitated and well positioned apartment within a good neighbourhood, is also a large contributing factor.
# - Size - The more space and rooms that the apartment contains, the higher the valuation.
# - Features - A fully equipped home with extras (such as a garage) are highly desirable.


# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew

plt.style.use(style='ggplot')

# Import the cleaned data from the cleaning part
df = pd.read_csv(
    'https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/cleaning/data_seloger_clean.csv')

'''''EXPLANATORY DATA ANALYSIS'''''

'''Quick analysis'''


# Create a simple preliminary EDA function
def quick_EDA_func(df):
    # Size of the data
    print("Initial data size is: {} ".format(df.shape))

    # Data types in the dataFrame
    print(df.dtypes)
    print("There are {} Numerical variables and {} Categorical variables".format(
        list(df.select_dtypes(include=[np.number]).shape)[1],
        list(df.select_dtypes(include=['object']).shape)[1]))

    # Describe the Target variable
    print(
        'Statistics of the target variable rental price:\n{} \n From the target variable we learn that the cheapest '
        'rent is of {}€ per month, the most expensive one is of {}€ per month and the mean is of {}€ per month'.format(
            df['rent'].describe(),
            df['rent'].min(), df['rent'].max(), round(df['rent'].mean())))

    # Plot histogram and probability of the rent, target variable
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.distplot(df['rent'], fit=norm);
    (mu, sigma) = norm.fit(df['rent'])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('Rent distribution\n')

    plt.subplot(1, 2, 2)
    res = stats.probplot(df['rent'], plot=plt)
    plt.suptitle('Target variable: rent')
    plt.show()

    print(
        'The Interactive Geographical Map provides additional information on the characteristics of each neighborhood')

    # Histogram of the number of listings per sector
    nbr = df[['rent', 'sector_name']].groupby('sector_name').count().sort_values(by='rent', ascending=False)
    nbr.reset_index(0, inplace=True)
    nbr.rename(columns={'rent': '# Apartments'}, inplace=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=nbr['sector_name'], y=nbr['# Apartments'], palette="Reds_r")
    plt.xlabel('\nSectors')
    plt.ylabel("Number of listings\n")
    plt.title("Number of listings per sector in Toulouse on SeLoger.com\n")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Histogram of the median rental price per sector
    price = df[['rent', 'sector_name']].groupby('sector_name').median().round().sort_values(by='rent', ascending=False)
    price.reset_index(0, inplace=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=price['sector_name'], y=price['rent'], palette="Blues_r")
    plt.xlabel('\nSectors')
    plt.ylabel('Median Rental Price (€)\n')
    plt.title("Median Rental Price per sector in Toulouse on SeLoger.com\n")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visual inspection between rental price and area of the property per sector
    sns.pairplot(x_vars=["area"], y_vars=["rent"], data=df, hue="sector_name", height=5)
    plt.title("Relation between Area and Rental Price per sector in Toulouse on seLoger.com\n")
    plt.show()


'''Correlation with the target variable'''


# Looking for correlations between different variables in the dataset,
# we find out the features that correlate the most to the rent of the apartment in the dataset
# and plot the correlation matrix as a heatmap.
# Note: when the variables are not normally distributed
# or the relationship between the variables is not linear (as is the case here),
# it is more appropriate to use the Spearman rank correlation method rather than the default Pearson's method.

# Create a function that return correlation matrix heatmaps
def heatmaps_func(df, method):
    # Set the style of the visualization
    sns.set(style="white")

    # Create a covariance matrix
    corrmat = df.corr(method=method)
    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 16))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corrmat, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},
                vmax=corrmat[corrmat != 1.0].max().max())
    plt.title('Correlation Matrix')
    plt.tight_layout()
    # plt.savefig('cormat.png')
    plt.show()

    # Correlation matrix of variables that have the highest correlation with 'rent'
    top_feature = corrmat[abs(corrmat['rent']) > 0.15].index
    top_corrmat = df[top_feature].corr(method=method)
    # Generate a mask the size of our covariance matrix
    top_mask = np.zeros_like(top_corrmat, dtype=np.bool)
    top_mask[np.triu_indices_from(top_mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(top_corrmat, mask=top_mask, cmap=cmap, center=0, annot=True, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                vmax=corrmat[corrmat != 1.0].max().max())
    plt.title('Correlation Matrix of top features')
    plt.tight_layout()
    # plt.savefig('cormat_top_feat.png')
    plt.show()

    print('Find the most important features relative to target')
    corrmat = df.corr(method='spearman')
    corrmat.sort_values(['rent'], ascending=False, inplace=True)
    return corrmat.rent


# Create a function that plot the relationship between 'rent'
# and the 9 variables that have the highest correlation with 'rent'
def regplot_top_feat_func(df):
    # Plot the relationship between 'rent' and the 6 variables that have the highest correlation with 'rent'
    topcorr = df.corr(method='spearman')['rent'].sort_values()[:-1]
    topcorr = topcorr.tail(9)
    fig, ax = plt.subplots(3, 3, figsize=(25, 12))
    ax = ax.ravel()
    j = 0
    for i in topcorr.index:
        sns.regplot(i, 'rent', ax=ax[j], lowess=True, data=df, color='C0', line_kws={"color": "C1"},
                    scatter_kws={"alpha": 0.2})
        ax[j].set_title("Rental Price vs {}".format(i))
        j += 1
    plt.tight_layout()
    # plt.savefig('Scatterplots.png')
    plt.show()


'''Visual check of the distribution of each feature'''


# For a efficient model we need to review one by one each feature to determine which transformation is needed.
# Then we will transform the numerical features first and then the categorical ones.

# Create a function that returns various plots for categorical and dummy features
def plot_cat_dum_feat_func(feat):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))
    sns.countplot(df[feat], ax=axs[0, 0])
    sns.catplot(x=feat, y='rent', data=df, ax=axs[0, 1])
    sns.boxplot(df[feat], df['rent'], ax=axs[1, 0])
    sns.barplot(df[feat], df['rent'], ax=axs[1, 1])
    plt.close(2)
    plt.tight_layout()
    plt.show()


# Create a function that returns various plots for numerical discrete features
def plot_discrete_feat_func(feat):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))
    sns.countplot(df[feat], palette="Blues_r", ax=axs[0, 0])
    sns.catplot(x=feat, y='rent', data=df, palette="Blues_r", ax=axs[0, 1])
    sns.boxplot(df[feat], df['rent'], palette="Blues_r", ax=axs[1, 0])
    sns.regplot(df[feat], df['rent'], color='darkblue',
                scatter_kws={"alpha": 0.2}, order=2,
                x_jitter=.1, ax=axs[1, 1])
    plt.close(2)
    plt.tight_layout()
    plt.show()


# Create a function that returns various plots for numerical continuous features
def plot_cont_feat_func(feat):
    fig = plt.subplots(ncols=1, figsize=(25, 6))

    # Distribution plot
    plt.subplot(1, 3, 1)
    sns.distplot(df[feat], fit=norm)
    (mu, sigma) = norm.fit(df['rent'])
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Distribution of {}'.format(feat))

    # Scatter plot
    plt.subplot(1, 3, 2)
    sns.regplot(df[feat], df['rent'], color='blue',
                scatter_kws={"alpha": 0.2}, order=2)
    corrcoef = df[feat].corr(df['rent'], method='spearman')
    plt.legend(['Spearman correlation: {:.3f}'.format(corrcoef)], loc='best')
    plt.title('Scatter Plot')

    # Probability plot
    plt.subplot(1, 3, 3)
    stats.probplot(df[feat], plot=plt)
    plt.show()


# Create a function that plots features depending on their type
def plot_all_feat_func(df):
    # 'agency': Name of the real estate agency in charge of the apartment.
    # Categorical feature
    plot_cat_dum_feat_func('agency')
    # - Here the real estate agency doesn't appear to have much of an impact on the rent.
    # - We will create a dummy variable whether or not there is an agency.

    # 'sector_no': Sector code of the apartment.
    # Categorical feature
    plot_cat_dum_feat_func('sector_no')
    # - Sectors have a contribution towards rent,
    # since we see slightly higher values for certain areas and lower values for others.
    # - Since it is categorical feature without order we will use One-Hot-Encoding to create dummy variables.

    # 'nbhd_no': Neighborhood code of the apartment.
    # Categorical feature
    plot_cat_dum_feat_func('nbhd_no')
    # - Neighborhoods have a contribution towards rent, since we see slightly higher values
    # for certain areas and lower values for others.
    # - Since it is categorical feature without order we will use One-Hot-Encoding to create dummy variables.

    # 'charges': Type of charges of the apartment.
    # Categorical feature
    plot_cat_dum_feat_func('charges')
    # - This feature does not hold much information since the numbers of observations for the class +CH' is too low.
    # - We will simply drop the feature.

    # 'fees': Fees charged to the tenant in euros set by the agency.
    # Numerical continuous feature
    plot_cont_feat_func('fees')
    # - Here we see a positive correlation with rent as the fees increase.
    # - This amount is calculated by the agency (if an agency) based on the the geographical location of the apartment
    # and its size in square meters of the apartment and not directly linked to the rent.
    # - We will drop this feature.

    # 'deposit': Deposit in euros.
    # Numerical continuous feature
    plot_cont_feat_func('deposit')
    # - Here we see a positive correlation with rent as the deposit increases.
    # - This relation seems to be normal as the deposit amount is based on the rent,
    # most of the time equal to one rent without charges.
    # Thus we cannot use this feature to predict the rent.
    # - We will drop this feature.

    # 'energy_rating': Energy performance diagnostic of the apartment.
    # Numerical continuous feature
    plot_cont_feat_func('energy_rating')
    # - Here we do not see a significant correlation with rent as the energy rating increases.
    # - We will test the feature for normality.

    # 'gas_rating': Greenhouse gas emission index of the apartment.
    # Numerical continuous feature
    plot_cont_feat_func('gas_rating')
    # - Here we do not see a significant correlation with rent as the gas rating increases.
    # - We will test the feature for normality.

    # 'area': Total area in square meters of the apartment.
    # Numerical continuous feature
    plot_cont_feat_func('area')
    # - Here we see a positive correlation with rent as the total area increases.
    # - We will test the feature for normality.

    # 'rooms': Total number of rooms in the apartment.
    # Numerical discrete feature
    plot_discrete_feat_func('rooms')
    # - We see a lot of houses with 1, 2 or 3 rooms, and a very low number of apartments with 6 or above.
    # - Here we see a positive correlation with rent as the total number of rooms increase.
    # - Since this is a continuous numeric feature, we will test the feature for normality.

    # 'entrance': Whether or not the apartment has an entrance room.
    # Dummy feature
    plot_cat_dum_feat_func('entrance')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'duplex': Whether or not it is a duplex.
    # Dummy feature
    plot_cat_dum_feat_func('duplex')
    # - Here it appears that duplex apartments are able to demand a higher average rent than ones that are not duplexes.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'livingroom': Whether or not it has a living-room.
    # Dummy feature
    plot_cat_dum_feat_func('livingroom')
    # - Here it appears that apartments with a living room are able to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'livingroom_area': Total living-room area in square meters.
    # Numerical continuous feature
    plot_cont_feat_func('livingroom_area')
    # - Here we see a positive correlation with rent as the livingroom area increases.
    # - We will test the feature for normality.

    # 'equipped_kitchen': Whether or it has an equipped kitchen.
    # Dummy feature
    plot_cat_dum_feat_func('equipped_kitchen')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'openplan_kitchen': Whether or not it has an open-plan kitchen.
    # Dummy feature
    plot_cat_dum_feat_func('openplan_kitchen')
    # - Here it appears that apartments with an open-plan kitchen are able
    # to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'bedrooms': Total number of bedrooms.
    # Numerical discrete feature
    plot_discrete_feat_func('bedrooms')
    # - We see a lot of apartments with 1, 2 and 3 bedrooms, and a very low number of apartments with 4 or above.
    # - Here we see a positive correlation with rent as the number of bedrooms increase.
    # - Since this is a continuous numeric feature, we will test the feature for normality.

    # 'bathrooms': Whether or not it has a bathroom.
    # Dummy feature
    plot_cat_dum_feat_func('bathrooms')
    # - Here it appears that apartments with at least one bathroom are able
    # to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'shower_rooms': Whether or not it has a shower-room.
    # Dummy feature
    plot_cat_dum_feat_func('shower_rooms')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'toilets': Total number of toilets.
    # Numerical discrete feature
    plot_discrete_feat_func('toilets')
    # - We see a lot of apartments with 1  toilet and a very low number with 2 toilets or more.
    # - Here we see a strong positive correlation with rent as the number of toilets increase.
    # - Since this is a continuous numeric feature, we will test the feature for normality.

    # 'separate_toilet': Whether or not it has separate toilet.
    # Dummy feature
    plot_cat_dum_feat_func('separate_toilet')
    # - Here it appears that apartments with at separate toilet are able
    # to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'balcony': Total number of balconies.
    # Numerical discrete feature
    plot_discrete_feat_func('balcony')
    # - We see a lot of apartments with 0 or 1 balcony and a very low number with 2 or more.
    # - Here we see a positive correlation with rent as the number of balconies increase.
    # - Since this is a continuous numeric feature, we will test the feature for normality.

    # 'terraces': Total number of terraces.
    # Numerical discrete feature
    plot_discrete_feat_func('terraces')
    # - We see a lot of apartments with 0 or 1 terraces and a very low number with 2 or more.
    # - Here we see a positive correlation with rent as the number of terraces increase.
    # - Since this is a continuous numeric feature, we will test the feature for normality.

    # 'wooden_floor': Whether or not it has some wooden floor.
    # Dummy feature
    plot_cat_dum_feat_func('wooden_floor')
    # - Here it appears that apartments with wooden floor are able to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'fireplace': Whether or not it has a fireplace.
    # Dummy feature
    plot_cat_dum_feat_func('fireplace')
    # - Here it appears that apartments with a fireplace are able to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'storage': Whether or not there is some storage within the apartment.
    # Dummy feature
    plot_cat_dum_feat_func('storage')
    # - Here it appears that apartments with some storage are able
    # to demand a slightly higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'heating': Type of heating system.
    # Categorical feature
    plot_cat_dum_feat_func('heating')
    # - Here it appears that apartments with central or gas heating system are able
    # to demand a slightly higher average rent than other ones.
    # - Since it is categorical feature without order we will use One-Hot-Encoding to create dummy variables.

    # 'parking': Total number of parking places.
    # Numerical discrete feature
    plot_discrete_feat_func('parking')
    # - We see a lot of apartments with 0 or 1 parking place and a very low number with 2 or more.
    # - Here we see a positive correlation with rent as the number of terraces increase.
    # - Since this is a continuous numeric feature, we will test the feature for normality.

    # 'cellar': Whether or not it has a cellar.
    # Dummy feature
    plot_cat_dum_feat_func('cellar')
    # - Here it appears that apartments with a cellar are able to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'pool': Whether or not the residence of the apartment has a pool.
    # Dummy feature
    plot_cat_dum_feat_func('pool')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'apt_flr_nb': Floor number of the apartment.
    # Numerical discrete feature
    plot_discrete_feat_func('apt_flr_nb')
    # - We see a lot of apartments at the 1st, 2nd or 3rd floor and a very low number at the 4th floor or higher.
    # - Here we see a positive correlation with rent as the number of the floor increases.
    # - Since this is a continuous numeric feature, we will test the feature for normality.

    # 'furnished': Whether or not it is furnished.
    # Dummy feature
    plot_cat_dum_feat_func('furnished')
    # - Here it appears that furnished apartments are able to demand a higher average rent than non-furnished ones.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'renovated': Whether or not it has been renovated recently.
    # Dummy variable
    plot_cat_dum_feat_func('renovated')
    # - Here it appears that recently renovated apartments are able
    # to demand a higher average rent than non-renovated ones.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'elevator': Whether or not there is an elevator in the building of the apartment.
    # Dummy feature
    plot_cat_dum_feat_func('elevator')
    # - Here it appears that apartments with an elevator in the building are able
    # to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'intercom': Whether or not it has an intercom.
    # Dummy variable
    plot_cat_dum_feat_func('intercom')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'digital_code': Whether or not it has a digital code at the entrance of the building of the apartment.
    # Dummy feature
    plot_cat_dum_feat_func('digital_code')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'view': Whether or not it has a nice view.
    # Dummy feature
    plot_cat_dum_feat_func('view')
    # - Here it appears that apartments with a view are able to demand a higher average rent than ones without.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'caretaker': Whether or not it has a caretaker.
    # Dummy feature
    plot_cat_dum_feat_func('caretaker')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'reduced_mobility': Whether or not it is accessible to reduced mobility persons.
    # Dummy feature
    plot_cat_dum_feat_func('reduced_mobility')
    # - Here it appears that apartments accessible to reduced mobility persons able
    # to demand a higher average rent than ones that are not accessible to them.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'metro': Whether or not it is close to a metro station
    # Dummy feature
    plot_cat_dum_feat_func('metro')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'tram': Whether or not it is close to a tram station
    # Dummy feature
    plot_cat_dum_feat_func('tram')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.

    # 'bus': Whether or not it is close to a bus station
    # Dummy feature
    plot_cat_dum_feat_func('bus')
    # - We do not see a clear significant correlation with the rent.
    # - Since this is a dummy variable, we will leave it how it is.


# Run the EDA functions
quick_EDA_func(df)
heatmaps_func(df, 'spearman')
regplot_top_feat_func(df)

# plot_all_feat_func(df)


''''' FEATURE ENGINEERING'''''

'''Drop several features (discussed in the EDA part)'''


# Create a function that drops the selected features
def drop_feat_func(df):
    print("Data size before dropping the features is: {} ".format(df.shape))

    # List of features to OHE
    feat_to_drop = ['link', 'sector_name', 'nbhd_name', 'charges', 'deposit', 'fees']
    # Drop columns
    df.drop(feat_to_drop, axis=1, inplace=True)

    print('Features {} have been dropped'.format(feat_to_drop))
    print("Data size after dropping the features is: {} ".format(df.shape))
    return df


'''One-Hot-Encoding of multiple categorical variables (discussed in the EDA part)'''


# Create a One-Hot-Encoding function
def ohe_func(df):
    # List of features to OHE
    feat_to_ohe = ['sector_no', 'nbhd_no', 'heating']
    # One hot encode
    df = pd.get_dummies(df, columns=feat_to_ohe)

    print('Features {} One-Hot-Encoded'.format(feat_to_ohe))
    return df


# ## Transform some features to dummies (discussed in the EDA part)
# Create a function that transform some features to dummy a dummy
def to_dummy_func(df):
    # Transform to dummy
    df['agency_flag'] = [0 if x == 'None' else 1 for x in df['agency']]
    # Drop feature
    df.drop(['agency'], axis=1, inplace=True)

    print('Feature agency transformed to dummy variable')
    return df


'''Log transformation of the Target Variable (discussed in the EDA part)'''


# Skewed numeric variables are not desirable when using Machine Learning algorithms.
# The reason why we want to do this is move the models focus away from any extreme values,
# to create a generalised solution. We can tame these extreme values by transforming skewed features

# Create a function that log transform the target variable
def log_transform_target_feat_func(df):
    # Skewness and Kurtosis before transformation
    print("Skewness before transformation: %f" % df['rent'].skew())
    print("Kurtosis before transformation: %f" % df['rent'].kurt())
    # The distribution of the target variable is positively skewed,
    # meaning that the mode is always less than the mean and median.

    # log1p transformation
    df['rent'] = np.log1p(df.rent)
    print('Target variable log transformed')

    # Plot histogram and probability after log1p transformation
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.distplot(df['rent'], fit=norm);
    (mu, sigma) = norm.fit(df['rent'])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('Rent distribution')
    plt.subplot(1, 2, 2)
    res = stats.probplot(df['rent'], plot=plt)
    plt.suptitle('Target variable after transformation')
    plt.show()

    # Skewness and kurtosis after transformation
    print("Skewness after transformation: %f" % df['rent'].skew())
    print("Kurtosis after transformation: %f" % df['rent'].kurt())
    return df


# We can see from the skewness and the plot that it follows much more closely to the normal distribution now.
# This will help the algorithms work most reliably because we are now predicting a distribution that is well-known,
# i.e. the normal distribution.


'''Transform Skewed Features (skewness > 0.5)'''


# Create a function that transform skewed features
def transform_feat_func(df):
    # Features to check for skewness (numerical features)
    numeric_feats = ['energy_rating', 'gas_rating', 'area', 'rooms', 'livingroom_area', 'bedrooms',
                     'toilets', 'balcony', 'terraces', 'parking', 'apt_flr_nb']

    # Check the skew of all numerical features before transformation
    skewness_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewness_feats})
    print("Skew in numerical features:{}\n".format(skewness))

    # Select skewed features (skewness > 0.50)
    skewed = skewness[abs(skewness) > 0.50]
    skewed = skewed.dropna()
    print(
        "There are {} skewed numerical features to Box Cox transform out of the {} initial numerical features.".format(
            skewed.shape[0], skewness.shape[0]))

    # Box Cox Transformation of skewed features
    skewed_features = skewed.index
    lam = 0.15  # set λ to 0.15
    for feat in skewed_features:
        # df[feat] += 1
        df[feat] = boxcox1p(df[feat], lam)

    # Check the skew of all numerical features after transformation
    skewness_feats_after = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("Skew in numerical features:")
    skewness_after = pd.DataFrame({'Skew': skewness_feats_after})
    print("Skew in numerical features after Box Cox transformation:{}\n".format(skewness_after))

    # Select the skewed features (skewness > 0.50)
    skewed_after = skewness_after[abs(skewness_after) > 0.50]
    skewed_after = skewed_after.dropna()
    print(
        "There are {} remaining skewed numerical features after Box Cox transform out of the {} skewed numerical features before transformation.".format(
            skewed_after.shape[0], skewed.shape[0]))
    return df


# We can see that a lot of parameters remained skewed due probably to features with lots of 0 values.


'''Feature Selection: Assess for multi-collinearity of features'''


# Reduce multi-collinearity as some of our top predictors are highly correlated:
# 'area', 'rooms' and 'bedrooms'
# neighborhood and sector features

def drop_feat_reduce_multicollinearity_func(df, method):
    # Check heatmaps before dropping the columns
    heatmaps_func(df, method)

    print("Data size before dropping the features is: {} ".format(df.shape))

    # List of features to OHE
    feat_to_drop = ['rooms', 'bedrooms',
                    'sector_no_sector1', 'sector_no_sector2', 'sector_no_sector3',
                    'sector_no_sector4', 'sector_no_sector5', 'sector_no_sector6']
    # Drop columns
    df.drop(feat_to_drop, axis=1, inplace=True)

    print('Features {} have been One-Hot-Encoded'.format(feat_to_drop))
    print("Data size after dropping the features is: {} ".format(df.shape))

    # Check heatmaps before dropping the columns
    heatmaps_func(df, method)
    return df


# (Optional)'''Feature Selection: Remove features with 0.0 importance'''


# (Optional) '''Scaling features'''
# In most cases, the numerical features of the dataset do not have a certain range and they differ from each other.
# In real life, it is nonsense to expect rental price and the size of the apartment SqM to have the same range.
# Scaling solves this problem. The continuous features become identical in terms of the range, after a scaling process.
# This process is not mandatory for many algorithms, but it might be still nice to apply.
# However, the algorithms based on distance calculations
# such as k-NN or k-Means need to have scaled continuous features as model input.
# Basically, there are two common ways of scaling: Normalization & Standardization


# Create a feature engineering function
def feat_eng_func(df):
    df = drop_feat_func(df)
    df = ohe_func(df)
    df = to_dummy_func(df)
    df = log_transform_target_feat_func(df)
    df = transform_feat_func(df)
    df = drop_feat_reduce_multicollinearity_func(df, 'spearman')
    return df


df = feat_eng_func(df)


# ### Export the file
df.to_csv('data_for_modeling.csv', index=False)
print("Data exported")
