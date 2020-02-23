#!/usr/bin/env python
# coding: utf-8

# ### Explanatory Data Analysis ### #
# 
# First of all, to guide our feature engineering, we would like to know what factors add value to an apartment?
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
from scipy.stats import norm, skew #for some statistics

plt.style.use(style='ggplot')




# Import the cleaned data from the cleaning part
df = pd.read_csv('https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/cleaning/data_seloger_clean.csv')

# check the size of the data
print("Initial data size is: {} ".format(df.shape))

# first visual inspection of the df
df.head()


# In[4]:


#  columns in the dataset:
print(df.columns)


# ## 3.1 Data Preparation

# ### 3.1.1 Remove duplicates

# As discussed in the previous part we found some duplicate entries based on the column 'link'
# We need to delete duplicate entries in the dataset as they would affect our analysis as our learning algorithm would learn from incorrect data. 

# In[5]:


# Finding out duplicates
uniqueRows = len(set(df.link))
totalRows = len(df.link)
duplicateRows = totalRows - uniqueRows
print('There are {} duplicates'.format(duplicateRows))


# In[6]:


# dropping duplicate values 
df = df.drop_duplicates(subset='link', keep="first")

print("Data size after dropping duplicate values is: {} ".format(df.shape)) 


# ### 3.1.2 Remove Outliers
# 
# Outliers can be a Data Scientists nightmare.
# - By definition, an outlier is something that is outside of the expected response. How far you're willing to consider something to be an outlier, is down to the individual and the problem.
# - From this definition, this outlier will therefore sit way outside of the distribution of data points. Hence, this will skew the distribution of the data and potential calculations.
# 
# Here we are trying to identify the outliers in the data which we cannot use in the model because it skews the analysis towards values that are unlikely:

# In[7]:


# Let's explore these outliers
fig, ax = plt.subplots()
ax.scatter(df['area'], df['rent'],color='blue')
plt.ylabel('rent', fontsize=13)
plt.xlabel('area', fontsize=13)
plt.show()


# We identified one outlier, we will then remove it:

# In[8]:


# Cleaning the dataset from its outliers
df = df.drop(df[(df['area']<40) & (df['rent']>1500)].index)


# Finally, let's check the data after removing the outlier:

# In[9]:


#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(df['area'], df['rent'],color='blue')
plt.ylabel('rent', fontsize=13)
plt.xlabel('area', fontsize=13)
plt.show()


# ### 3.1.3 Finding and filling Missing Values
# 
# Missing values are the Data Scientists other nightmare. They can mean multiple things:
# - A missing value may be the result of an error during the production of the dataset. Depending on where the data comes from, this could be:
#     - a human error
#     - a machinery error
# - A missing value in some cases, may just mean a that a 'zero' should be present. In which case, it can be replaced by a 0. The data description provided helps to address situations like these.
# - Otherwise, missing values represent no information. Therefore, does the fact that you don't know what value to assign an entry, mean that filling it with a 'zero' is always a good fit?
# 
# Some algorithms do not like missing values. Some are capable of handling them, but others are not. Therefore since we are using a variety of algorithms, it's best to treat them in an appropriate way. If you have missing values, you have two options:
# - Delete the entire row
# - Fill the missing entry with an imputed value
# 
# In order to treat this dataset we will cycle through each feature with missing values and treat them individually based on the data description, or our judgement. Let's first take identify features with missing values and their share of missing values:

# In[10]:


# Graph of the missing values
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
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_dataSubset=missing_data[missing_data['Total']>0]
print(missing_dataSubset)


# Through reference of the data description, this gives guidance on how to treat missing values for some columns. For ones where guidance isn't clear enough, we have to use intuition as explained below.

# We can clearly see columns 'postal_code', 'orientation', 'bldg_flr_nb' and 'construction_year' having a lot of values that are missing (Nan% greater than 50). 
# We know that for those columns, missing values mean that we do not have the information. We decide to drop those columns as there are too many missing values:

# In[11]:


df.drop(['postal_code','orientation','bldg_flr_nb','construction_year'], axis=1, inplace=True)


# We impute the rest of the columns by proceeding sequentially:

# - nbhd_no / nbhd_name : According to data description missing values are due to a lack of information - Replacing missing data with the mode of 'neighborhood'
# - sector_no / sector_name : According to data description missing values are due to a lack of information - Replacing missing data with the appropriate sector according to the neighborhood the appartment is in.

# In[12]:


df['nbhd_no'] = df['nbhd_no'].fillna(df['nbhd_no'].mode()[0])

#Group by neighborhood and fill in missing value by the mode of neighborhood
df['nbhd_name'] = df.groupby('nbhd_no')['nbhd_name'].transform(lambda x: x.fillna(x.mode()[0]))
df['sector_no'] = df.groupby('nbhd_no')['sector_no'].transform(lambda x: x.fillna(x.mode()[0]))
df['sector_name'] = df.groupby('nbhd_no')['sector_name'].transform(lambda x: x.fillna(x.mode()[0]))


# - agency: According to data description missing values are due to the the fact that some appartment are not rented via a real estaste agency - Replacing missing data with 'None'

# In[13]:


df['agency'] = df['agency'].fillna('None')


# - provisions : According to data description missing values are, if rented via a real estate agency, due to a lack of information and since provisions of each appartment most likely have similar provisions to other appartments in its neighborhood we can fill in missing values by the median 'provisions' of the neighborhood if 'agency' different to 'None', else replace with 0 
# - fees : According to data description missing values are, if rented via a real estate agency, due to a lack of information and since fees of each appartment most likely have similar fees to other appartments in its neighborhood we can fill in missing values by the median 'fees' of the neighborhood if 'agency' different to 'None', else replace with 0

# In[14]:


# If no agency then 0
for col in ('provisions', 'fees'):
    df.loc[(df[col].isnull()) & (df.agency == 'None'), col] = 0

# If agency then group by neighborhood and fill in missing value by the median value of neighborhood
for col in ('provisions', 'fees'):
    df[col] = df.groupby('nbhd_no')[col].transform(lambda x: x.fillna(x.median()))


# - deposit: According to data description missing values are due to a lack of information and since the deposit of each appartment most likely have a similar deposit to other appartments in its neighborhood we can fill in missing values by the median 'deposit' of the neighborhood
# - energy_rating: According to data description missing values are due to a lack of information and since the area of each appartment most likely have a similar energy rating to other appartments in its neighborhood we can fill in missing values by the median 'energy_rating' of the neighborhood
# - gas_rating: According to data description missing values are due to a lack of information and since the gas rating of each appartment most likely have a similar gas rating to other appartments in its neighborhood we can fill in missing values by the median 'gas_rating' of the neighborhood
# - apt_flr_nb : According to data description missing values are due to a lack of information and since buildings in the same neighborhood most likely have the same total number of floors we can fill in missing values by the median 'apt_flr_nb' of the neighborhood
# - toilets: According to data description missing values are due to a lack of information and since the number of toilets of each appartment most likely have a similar number of toilets to other appartments in its neighborhood we can fill in missing values by the median 'toilets' of the neighborhood
# - area: According to data description missing values are due to a lack of information and since the area of each appartment most likely have a similar area to other appartments in its neighborhood we can fill in missing values by the median 'area' of the neighborhood

# In[15]:


#Group by neighborhood and fill in missing value by the median value of neighborhood
for col in ('deposit', 'energy_rating', 'gas_rating', 'apt_flr_nb', 'toilets', 'area'):
    df[col] = df.groupby('nbhd_no')[col].transform(lambda x: x.fillna(x.median()))


# - heating: According to data description missing values are due to a lack of information. We can fill in missing values by the mode 'heating' of the neighborhood

# In[16]:


#Group by neighborhood and fill in missing value by the mode of the neighborhood
df['heating'] = df.groupby('nbhd_no')['heating'].transform(lambda x: x.fillna(x.mode()[0]))


# Finally we verify that we got rid of all missing values:

# In[17]:


# Final inspection of missing values
print('There are {} missing value(s) left'.format(sum(df.isnull().sum())))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
plt.show()


# Before going to next part, we will save a version of the cleaned dataset as it will be useful later for our map analysis but also to get a final table with predicted rent:

# In[18]:


df.to_csv('data_seloger_datapreparation_clean_part3.csv',index=False)


# Finally we drop the  features 'link' ,'sector_name', 'nbhd_name' as it is not usefull for our model.

# In[19]:


#check the numbers of samples and features
print("Data size before dropping link feature is: {} ".format(df.shape))

# Drop features
df.drop(['link','sector_name','nbhd_name'], axis=1, inplace=True)

#check again the data size after dropping the 'link' variable
print("Data size after dropping link feature is: {} ".format(df.shape)) 


# ## 3.2 Exploratory Data Analysis

# ### 3.2.1 Correlation matrix

# Now that missing values and outliers have been treated, we will analyse each feature in more detail. 
# This will give guidance on how to prepare this feature for modeling. We will analyse the features based on the different aspects of the apartment available in the dataset.

# Looking for correlations between different variables in the dataset, we find out the features that correlate the most to the rent of the apartment in the dataset and plot the correlation matrix as a heatmap.
# Note: when the variables are not normally distributed or the relationship between the variables is not linear (as is the case here), it is more appropriate to use the Spearman rank correlation method rather than the default Pearson's method.

# In[20]:


#Correlation map to see how features are correlated with SalePrice
corrmat = df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()


# In[21]:


#correlation matrix
corrmat = df.corr(method='spearman')
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat,vmax=.5, square=True, fmt='.2f', linewidths=.2, cmap="YlGnBu")
#sns.heatmap(corrmat,vmin=-1, vmax=1, center=0, square=True, cmap=sns.diverging_palette(20, 220, n=200))
plt.show()


# In[119]:


corrmat = df.corr(method='spearman')
top_feature = corrmat[abs(corrmat['rent']>0.20)].index
top_corrmat = df[top_feature].corr(method='spearman')

plt.subplots(figsize =(12, 8)) 
sns.heatmap(top_corrmat, annot=True,cmap ="YlGnBu",linewidths = 0.1, 
            vmin=0.20, vmax=1, square=True,
            yticklabels = top_feature.values,xticklabels = top_feature.values)
plt.show()


# In[79]:


print('Find the most important features relative to target')
corrmat = df.corr(method='spearman')
corrmat.sort_values(['rent'],ascending = False, inplace = True)
corrmat.rent


# Using this correlation matrix, we are able to visualise the raw highly influencing factors on rent. such as the number of rooms, the number of bedrooms and also the total area size in square meters.
# 
# However it also seems that those factors are highly correlated between each other we can cause later some overfitting to our model as those features will bring the same information to our model. Indeed, we can easily imagine that a bigger apartment will have more rooms and more bedrooms to divide the extra space.   
# 
# In conclusion, there are many redundant features and this is bad, if we add redundant information to our model it keeps learning the same thing again and again and that doesn’t help, whenever it’s possible we want to remove redundant features.

# ### 3.2.1 Feature Engineering
# For a performant model we need to review one by one each feature to determine which transformation is needed.
# Then we will transform the numerical features first and then the categorical ones.

# #### 3.2.1.1 Visual check of the distribution of each feature

# In[23]:


#  columns in the dataset:
print(df.columns)


# #### agency: Name of the real estate agency in charge of the appartment.
# Categorical feature

# In[24]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(50,6))

sns.countplot(df['agency'], ax=axs[0]) 
sns.catplot(x="agency", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['agency'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['agency'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here the agency doesn't appear to have much of an impact on the rent.
# - We will create a dummy variable whether or not there is an agency.

# #### sector_no: Sector code of the appartment. 
# Categorical feature

# In[25]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['sector_no'], ax=axs[0]) 
sns.catplot(x="sector_no", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['sector_no'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['sector_no'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Sectors have a contribution towards rent, since we see slightly higher values for certain areas and lower values for others.
# - Since it is categorical feature without order we will use One-Hot-Encoding to create dummy variables.

# #### nbhd_no: Neighborhood code of the appartment. 
# Categorical feature

# In[26]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['nbhd_no'], ax=axs[0]) 
sns.catplot(x="nbhd_no", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['nbhd_no'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['nbhd_no'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Neighboroods have a contribution towards rent, since we see slightly higher values for certain areas and lower values for others.
# - Since it is categorical feature without order we will use One-Hot-Encoding to create dummy variables.

# #### charges: Type of charges of the appartment. 
# Categorical feature

# In[27]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['charges'], ax=axs[0]) 
sns.catplot(x="charges", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['charges'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['charges'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - This feature does not hold much information since the numbers of observations for the class +CH' is too low.
# - We will simply drop the feature.

# #### provisions: Provisions for charges with annual adjustment in euros set by the agency. 
# Numerical feature 

# In[28]:


# Graphs for numerical features 
fig, axs = plt.subplots(ncols=2,figsize=(20,6))

sns.distplot(df['provisions'] , fit=norm, ax=axs[0]) 
sns.scatterplot(df['provisions'], df['rent'],color='blue', ax=axs[1]);


# - Here we see a positive correlation with rent as the provisions increase.
# - We will test the feature for normality.

# #### fees: Fees charged to the tenant in euros set by the agency. 
# Numerical feature

# In[29]:


# Graphs for numerical features 
fig, axs = plt.subplots(ncols=2,figsize=(20,6))

sns.distplot(df['fees'] , fit=norm, ax=axs[0]) 
sns.scatterplot(df['fees'], df['rent'],color='blue', ax=axs[1]);


# - Here we see a positive correlation with rent as the fees increase.
# - This amount is calculated by the agency (if an agency) based on the the geographical location of the appartment and its size in square meters of the appartment and not direclty linked to the rent. We will keep this feature while keeping in mind its supposed correlation with area and neighborhood
# - We will test the feature for normality.

# #### deposit: Deposit in euros. 
# Numerical feature

# In[30]:


# Graphs for numerical features 
fig, axs = plt.subplots(ncols=2,figsize=(20,6))

sns.distplot(df['deposit'] , fit=norm, ax=axs[0]) 
sns.scatterplot(df['deposit'], df['rent'],color='blue', ax=axs[1]);


# - Here we see a positive correlation with rent as the deposit increases.
# - This relation seems to be normal as the deposit amount is based on the rent, most of the time equal to one rent without charges. Thus we cannot use this feature to predict the rent.
# - We will drop this feature. 

# #### energy_rating: Energy performance diagnostic of the appartment. 
# Numerical feature

# In[31]:


# Graphs for numerical features 
fig, axs = plt.subplots(ncols=2,figsize=(20,6))

sns.distplot(df['energy_rating'] , fit=norm, ax=axs[0]) 
sns.scatterplot(df['energy_rating'], df['rent'],color='blue', ax=axs[1]);


# - Here we do not see a significant correlation with rent as the energy rating increases. 
# - We will test the feature for normality.

# #### gas_rating: Greenhouse gas emission index of the appartment. 
# Numerical feature

# In[32]:


# Graphs for numerical features 
fig, axs = plt.subplots(ncols=2,figsize=(20,6))

sns.distplot(df['gas_rating'] , fit=norm, ax=axs[0]) 
sns.scatterplot(df['gas_rating'], df['rent'],color='blue', ax=axs[1]);


# - Here we do not see a significant correlation with rent as the gas rating increases. 
# - We will test the feature for normality.

# #### area: Total area in square meters of the appartement. 
# Numerical feature

# In[33]:


# Graphs for numerical features 
fig, axs = plt.subplots(ncols=2,figsize=(20,6))

sns.distplot(df['area'] , fit=norm, ax=axs[0]) 
sns.scatterplot(df['area'], df['rent'],color='blue', ax=axs[1]);


# - Here we see a positive correlation with rent as the total area increases. 
# - We will test the feature for normality.

# #### rooms: Total number of rooms in the appartment. 
# Numerical feature

# In[34]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['rooms'], ax=axs[0]) 
sns.catplot(x="rooms", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['rooms'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['rooms'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We see a lot of houses with 1, 2 or 3 rooms, and a very low number of appartments with 6 or above.
# - Here we see a positive correlation with rent as the total number of rooms increase.
# - Since this is a continuous numeric feature, we will test the feature for normality.

# #### entrance: Whether or not the appartment has an entrance room.
# Dummy feature

# In[35]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['entrance'], ax=axs[0]) 
sns.catplot(x="entrance", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['entrance'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['entrance'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent. 
# - Since this is a dummy variable, we will leave it how it is.

# #### duplex: Whether or not it is a duplex. 
# Dummy feature

# In[36]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['duplex'], ax=axs[0]) 
sns.catplot(x="duplex", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['duplex'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['duplex'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that duplex appartments are able to demand a higher average rent than ones that are not duplexes. 
# - Since this is a dummy variable, we will leave it how it is.

# #### livingroom: Whether or not it has a living-room.
# Dummy feature

# In[37]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['livingroom'], ax=axs[0]) 
sns.catplot(x="livingroom", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['livingroom'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['livingroom'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with a living room are able to demand a higher average rent than ones without. 
# - Since this is a dummy variable, we will leave it how it is.

# #### livingroom_area: Total living-room area in square meters. 
# Numerical feature

# In[38]:


# Graphs for numerical features 
fig, axs = plt.subplots(ncols=2,figsize=(20,6))

sns.distplot(df['livingroom_area'] , fit=norm, ax=axs[0]) 
sns.scatterplot(df['livingroom_area'], df['rent'],color='blue', ax=axs[1]);
#sns.boxplot(df['livingroom_area'], df['rent'], ax=axs[2]); #.set(ylim=(0,800000))


# - Here we see a positive correlation with rent as the livingroom area increases.
# - We will test the feature for normality.

# #### equipped_kitchen: Whether or it has an equipped kitchen. 
# Dummy feature

# In[39]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['equipped_kitchen'], ax=axs[0]) 
sns.catplot(x="equipped_kitchen", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['equipped_kitchen'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['equipped_kitchen'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent. 
# - Since this is a dummy variable, we will leave it how it is.

# #### openplan_kitchen: Whether or not it has an open-plan kitchen. 
# Dummy feature

# In[40]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['openplan_kitchen'], ax=axs[0]) 
sns.catplot(x="openplan_kitchen", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['openplan_kitchen'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['openplan_kitchen'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with an open-plan kitchen are able to demand a higher average rent than ones without. 
# - Since this is a dummy variable, we will leave it how it is.

# #### bedrooms: Total number of bedrooms. 
# Numerical feature

# In[41]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['bedrooms'], ax=axs[0]) 
sns.catplot(x="bedrooms", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['bedrooms'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['bedrooms'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We see a lot of appartments with 1, 2 and 3 bedrooms, and a very low number of appartments with 4 or above.
# - Here we see a positive correlation with rent as the number of bedrooms increase.
# - Since this is a continuous numeric feature, we will test the feature for normality.

# #### bathrooms: Whether or not it has a bathroom. 
# Dummy feature

# In[42]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['bathrooms'], ax=axs[0]) 
sns.catplot(x="bathrooms", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['bathrooms'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['bathrooms'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with at least one bathroom are able to demand a higher average rent than ones without.
# - Since this is a dummy variable, we will leave it how it is.

# #### shower_rooms: Whether or not it has a shower-room.
# Dummy feature

# In[43]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['shower_rooms'], ax=axs[0]) 
sns.catplot(x="shower_rooms", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['shower_rooms'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['shower_rooms'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent. 
# - Since this is a dummy variable, we will leave it how it is.

# #### toilets: Total number of toilets.
# Numerical feature

# In[44]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['toilets'], ax=axs[0]) 
sns.catplot(x="toilets", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['toilets'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['toilets'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We see a lot of appartments with 1  toilet and a very low number with 2 toilets or more. 
# - Here we see a strong positive correlation with rent as the number of toilets increase. 
# - Since this is a continuous numeric feature, we will test the feature for normality.

# #### separate_toilet: Whether or not it has separate toilet. 
# Dummy feature

# In[45]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['separate_toilet'], ax=axs[0]) 
sns.catplot(x="separate_toilet", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['separate_toilet'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['separate_toilet'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with at separate toilet are able to demand a higher average rent than ones without. 
# - Since this is a dummy variable, we will leave it how it is.

# #### balcony: Total number of balconies. 
# Dummy feature

# In[46]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['balcony'], ax=axs[0]) 
sns.catplot(x="balcony", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['balcony'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['balcony'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We see a lot of appartments with 0 or 1 balcony and a very low number with 2 or more. 
# - Here we see a positive correlation with rent as the number of balconies increase. 
# - Since this is a continuous numeric feature, we will test the feature for normality.

# #### terraces: Total number of terraces.
# Dummy feature

# In[47]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['terraces'], ax=axs[0]) 
sns.catplot(x="terraces", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['terraces'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['terraces'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We see a lot of appartments with 0 or 1 terraces and a very low number with 2 or more.
# - Here we see a positive correlation with rent as the number of terraces increase.
# - Since this is a continuous numeric feature, we will test the feature for normality.

# #### wooden_floor: Whether or not it has some wooden floor. 
# Dummy feature

# In[48]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['wooden_floor'], ax=axs[0]) 
sns.catplot(x="wooden_floor", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['wooden_floor'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['wooden_floor'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with wooden floor are able to demand a higher average rent than ones without.
# - Since this is a dummy variable, we will leave it how it is.

# #### fireplace: Whether or not it has a fireplace. 
# Dummy feature

# In[49]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['fireplace'], ax=axs[0]) 
sns.catplot(x="fireplace", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['fireplace'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['fireplace'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with a fireplace are able to demand a higher average rent than ones without.
# - Since this is a dummy variable, we will leave it how it is.

# #### storage: Whether or not there is some storage within the appartment. 
# Dummy feature

# In[50]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['storage'], ax=axs[0]) 
sns.catplot(x="storage", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['storage'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['storage'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with some storage are able to demand a slightly higher average rent than ones without.
# - Since this is a dummy variable, we will leave it how it is.

# #### heating: Type of heating system. 
# Categorical feature

# In[51]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['heating'], ax=axs[0]) 
sns.catplot(x="heating", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['heating'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['heating'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with central or gas heating system are able to demand a slightly higher average rent than other ones.
# - Since it is categorical feature without order we will use One-Hot-Encoding to create dummy variables.

# #### parking: Total number of parking places. 
# Numerical feature

# In[52]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['parking'], ax=axs[0]) 
sns.catplot(x="parking", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['parking'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['parking'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We see a lot of appartments with 0 or 1 parking place and a very low number with 2 or more.
# - Here we see a positive correlation with rent as the number of terraces increase.
# - Since this is a continuous numeric feature, we will test the feature for normality.

# #### cellar: Whether or not it has a cellar. 
# Dummy feature

# In[53]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['cellar'], ax=axs[0]) 
sns.catplot(x="cellar", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['cellar'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['cellar'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with a cellar are able to demand a higher average rent than ones without.
# - Since this is a dummy variable, we will leave it how it is.

# #### pool: Whether or not the building of the appartment has a pool access. 
# Dummy feature

# In[54]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['pool'], ax=axs[0]) 
sns.catplot(x="pool", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['pool'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['pool'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent.
# - Since this is a dummy variable, we will leave it how it is.

# #### apt_flr_nb: Floor number of the appartment. 
# Numerical feature

# In[55]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['apt_flr_nb'], ax=axs[0]) 
sns.catplot(x="apt_flr_nb", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['apt_flr_nb'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['apt_flr_nb'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We see a lot of appartments at the 1st, 2nd or 3rd floor and a very low number at the 4th floor or higher.
# - Here we see a positive correlation with rent as the number of the floor increases.
# - Since this is a continuous numeric feature, we will test the feature for normality.

# #### furnished: Whether or not it is furnished. 
# Dummy feature

# In[56]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['furnished'], ax=axs[0]) 
sns.catplot(x="furnished", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['furnished'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['furnished'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that furnished appartments are able to demand a higher average rent than non-furnished ones.
# - Since this is a dummy variable, we will leave it how it is.

# #### renovated: Whether or not it has been renovated recently. 
# Dummy variable

# In[57]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['renovated'], ax=axs[0]) 
sns.catplot(x="renovated", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['renovated'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['renovated'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that recently renovated appartments are able to demand a higher average rent than non-renovated ones.
# - Since this is a dummy variable, we will leave it how it is.

# #### elevator: Whether or not there is an elevator in the building of the appartment. 
# Dummy feature

# In[58]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['elevator'], ax=axs[0]) 
sns.catplot(x="elevator", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['elevator'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['elevator'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with an elevator in the building are able to demand a higher average rent than ones without.
# - Since this is a dummy variable, we will leave it how it is.

# #### intercom: Whether or not it has an intercom. 
# Dummy variable

# In[59]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['intercom'], ax=axs[0]) 
sns.catplot(x="intercom", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['intercom'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['intercom'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent.
# - Since this is a dummy variable, we will leave it how it is.

# #### digital_code: Whether or not it has a digital code at the entrance of the building of the appartment. 
# Dummy feature

# In[60]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['digital_code'], ax=axs[0]) 
sns.catplot(x="digital_code", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['digital_code'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['digital_code'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent.
# - Since this is a dummy variable, we will leave it how it is.

# #### view: Whether or not it has a nice view. 
# Dummy feature

# In[61]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['view'], ax=axs[0]) 
sns.catplot(x="view", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['view'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['view'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments with a view are able to demand a higher average rent than ones without.
# - Since this is a dummy variable, we will leave it how it is.

# #### caretaker: Whether or not it has a caretaker. 
# Dummy feature

# In[62]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['caretaker'], ax=axs[0]) 
sns.catplot(x="caretaker", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['caretaker'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['caretaker'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent.
# - Since this is a dummy variable, we will leave it how it is.

# #### reduced_mobility: Whether or not it is accessible to reduced mobility persons. 
# Dummy feature

# In[63]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['reduced_mobility'], ax=axs[0]) 
sns.catplot(x="reduced_mobility", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['reduced_mobility'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['reduced_mobility'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - Here it appears that appartments accessible to reduced mobility persons able to demand a higher average rent than ones that are not accessible to them.
# - Since this is a dummy variable, we will leave it how it is.

# #### metro: Whether or not it is close to a metro station
# Dummy feature

# In[64]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['metro'], ax=axs[0]) 
sns.catplot(x="metro", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['metro'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['metro'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent.
# - Since this is a dummy variable, we will leave it how it is.

# #### tram: Whether or not it is close to a tram station
# Dummy feature

# In[65]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['tram'], ax=axs[0]) 
sns.catplot(x="tram", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['tram'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['tram'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent.
# - Since this is a dummy variable, we will leave it how it is.

# #### bus: Whether or not it is close to a bus station
# Dummy feature

# In[66]:


# Graphs for categorical/dummy features 
fig, axs = plt.subplots(ncols=4,figsize=(25,6))

sns.countplot(df['bus'], ax=axs[0]) 
sns.catplot(x="bus", y="rent", data=df, ax=axs[1]) 
sns.boxplot(df['bus'], df['rent'], ax=axs[2]) #.set(ylim=(0,800000))
sns.barplot(df['bus'], df['rent'], ax=axs[3]); #.set(ylim=(0,800000))


# - We do not see a clear significant correlation with the rent.
# - Since this is a dummy variable, we will leave it how it is.

# #### 3.2.1.2 Drop some features 
# As discussed above we will drop the features 'charges' and 'deposit' using .drop():

# In[67]:


print("Data size before dropping the features is: {} ".format(df.shape)) 

# Features to drop: charges deposit
df.drop(['charges','deposit'], axis=1, inplace=True)

print("Data size after dropping the features is: {} ".format(df.shape)) 


# #### 3.2.1.3 One-Hot-Encoding of multiple categorical variables
# As discussed above we will create dummy variables via One-Hot-Encoding the features 'sector' and 'neighborhood', 'heating'

# In[68]:


# List of features to OHE
feat_to_ohe=['sector_no','nbhd_no','heating']

# One hot encode
df = pd.get_dummies(df, columns=feat_to_ohe)


# #### 3.2.1.4 Create dummy variables
# As discussed above we will create a flag from the feature 'agency'

# In[69]:


# Transform to dummy variable: agency 
df['agency_flag'] = [0 if x =='None' else 1 for x in df['agency']] 

# Features to drop: agency
df.drop(['agency'], axis=1, inplace=True) 


# ### 3.2.2 Distribution of the Target Variable
# 
# In regression we are predicting a continuous number. Therefore, it is always useful to check the distribution of the target variable when building a regression model as Machine Learning algorithms work well with features that are normally distributed, a distribution that is symmetric and has a characteristic bell shape. 
# If features are not normally distributed, you can transform them using clever statistical methods.
# 
# So First, let's check the target variable rent.

# In[70]:


# Describe the Target variable
print(df['rent'].describe())

print('\n From the target variable we learn that the cheapest rent is of {}€ per month, the most expensive one is of {}€ per month and the mean is of {}€ per month'.format(df['rent'].min(),df['rent'].max(),round(df['rent'].mean())))


# In[71]:


# Plot histogram and probability
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(df['rent'] , fit=norm);
(mu, sigma) = norm.fit(df['rent'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Rent distribution')
plt.subplot(1,2,2)
res = stats.probplot(df['rent'], plot=plt)
plt.suptitle('Before transformation')

#skewness and kurtosis
print("Skewness before transformation: %f" % df['rent'].skew())
print("Kurtosis before transformation: %f" % df['rent'].kurt())


# The distribution of the target variable is positively skewed, meaning that the mode is always less than the mean and median.
# 
# In order to transform this variable into a distribution that looks closer to the black line shown above, we can use the numpy function log1p which applies log(1+x) to all elements within the feature.

# In[72]:


# Apply log1p transformation
df['rent'] = np.log1p(df.rent)


# In[73]:


# Plot histogram and probability after transformation
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(df['rent'] , fit=norm);
(mu, sigma) = norm.fit(df['rent'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Rent distribution')
plt.subplot(1,2,2)
res = stats.probplot(df['rent'], plot=plt)
plt.suptitle('After transformation')

#skewness and kurtosis
print("Skewness after transformation: %f" % df['rent'].skew())
print("Kurtosis after transformation: %f" % df['rent'].kurt())


# We can see from the skewness and the plot that it follows much more closely to the normal distribution now. This will help the algorithms work most reliably because we are now predicting a distribution that is well-known, i.e. the normal distribution.

# ### 3.2.3 Treating Skewed Features
# As said earlier, skewed numeric variables are not desirable when using Machine Learning algorithms. The reason why we want to do this is move the models focus away from any extreme values, to create a generalised solution. We can tame these extreme values by transforming skewed features.

# In[74]:


# Check for skewness the following features: 'provisions', 'fees', 'energy_rating', 'gas_rating', 'area', 'rooms', 'livingroom_area', 'bedrooms', 'toilets', 'balcony', 'terraces', 'parking', 'apt_flr_nb'
numeric_feats = ['provisions', 'fees', 'energy_rating', 'gas_rating', 'area', 'rooms', 'livingroom_area', 'bedrooms', 'toilets', 'balcony', 'terraces', 'parking', 'apt_flr_nb']

# Check the skew of all numerical features
skewness_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("Skew in numerical features:")
skewness = pd.DataFrame({'Skew' :skewness_feats})
skewness


# Clearly, we have a few positive skewing features. 
# We will now transform the features with skew > 0.5 to follow more closely the normal distribution.
# 
# We are using the Box-Cox transformation to transform non-normal variables into a normal shape. Normality is an important assumption for many statistical techniques; if your data isn't normal, applying a Box-Cox means that you are able to run a broader number of tests.
# We use the scipy function boxcox1p which computes the Box-Cox transformation of 1+x.
# Note that setting λ=0 is equivalent to log1p used above for the target variable.

# In[75]:


# Select the skewed features (skewness > 0.50)
skewed = skewness[abs(skewness) > 0.50]
skewed = skewed.dropna()
print("There are {} skewed numerical features to Box Cox transform out of the {} initial numerical features.".format(skewed.shape[0],skewness.shape[0]))

# Box Cox Transformation
from scipy.special import boxcox1p
skewed_features = skewed.index
lam = 0.15 # set λ to 0.15
for feat in skewed_features:
    #df[feat] += 1
    df[feat] = boxcox1p(df[feat], lam)
    
#df[skewed_features] = np.log1p(df[skewed_features])


# Observe now the correction.

# In[76]:


# Check the skew of all numerical features after Box Cox transform
skewness_feats_after = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("Skew in numerical features:")
skewness_after = pd.DataFrame({'Skew' :skewness_feats_after})
print(skewness_after.head(20))

# Select the skewed features (skewness > 0.50)
skewed_after = skewness_after[abs(skewness_after) > 0.50]
skewed_after = skewed_after.dropna()
print("There are {} remaining skewed numerical features after Box Cox transform out of the {} skewed numerical features before transformation.".format(skewed_after.shape[0],skewed.shape[0]))


# We can see that a lot of parameters remained skewed due probably to features with lots of 0 values.

# Now that our dataset is ready for modeling, we will export our dataset so we can prepare it from training, testing and prediction in our next part. 

# In[77]:


df.to_csv('data_seloger_EDA_part3.csv',index=False)


# In[78]:


df.columns


# Ref: - https://www.kaggle.com/agodwinp/stacking-house-prices-walkthrough-to-top-5/notebook
