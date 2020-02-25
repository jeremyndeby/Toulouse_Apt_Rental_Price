#!/usr/bin/env python
# coding: utf-8

# ### Model Selection ### #

# Import libraries
import numpy as np
import pandas as pd
import xgboost
from lightgbm import LGBMRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Import the data after feature engineering
df = pd.read_csv(
    'https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/EDA_feature_engineering/data_for_modeling.csv')


# ### Splitting data

# Set the target variable
y = df.rent.values
df.drop(['rent'], axis=1, inplace=True)

# Set predictors
X = df

# Split data (random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


# ### Define a rmse evaluation function
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# ### Create a comparison dictionary
model_score = {}


# ### Compare models

# ## Linear Regression
# Sometimes, simple models, outperformed more complex models like Random Forest and xgboost,
# especially on small datasets.
# Create instance
model_lr = linear_model.LinearRegression()
# Fit the model on the training set
model_lr.fit(X_train, y_train)
# Predict
y_pred_lr_train = model_lr.predict(X_train)
# Test
y_pred_lr_test = model_lr.predict(X_test)
# Results
print("LinearRegression Training set RMSE: : {:.4f}".format(rmse(y_train, y_pred_lr_train)))
print("LinearRegression Test set RMSE: : {:.4f}".format(rmse(y_test, y_pred_lr_test)))
print("LinearRegression Training set R^2: : {:.3f}".format(r2_score(y_train, y_pred_lr_train)))
print("LinearRegression Training set R^2: : {:.3f}".format(r2_score(y_test, y_pred_lr_test)))
# Add to the final comparison dictionary
model_score['linreg'] = rmse(y_test, y_pred_lr_test)

# ## Lasso
# Compute the cross-validation score with the default hyper-parameters
# Create instance
lassoCV = LassoCV(alphas=[1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2,
                          1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100])
# Fit the model on the training set
model_lasso = lassoCV.fit(X_train, y_train)
alpha_l = model_lasso.alpha_
print("Lasso best alpha :", alpha_l)
# Predict
y_pred_lasso_train = model_lasso.predict(X_train)
# Test
y_pred_lasso_test = model_lasso.predict(X_test)
# Results
print("Lasso Training set RMSE: : {:.4f}".format(rmse(y_train, y_pred_lasso_train)))
print("Lasso Test set RMSE: : {:.4f}".format(rmse(y_test, y_pred_lasso_test)))
# Add to the final comparison dictionary
model_score['lasso'] = rmse(y_test, y_pred_lasso_test)

coef = pd.Series(model_lasso.coef_, index=X_train.columns)
print("\nLasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
    sum(coef == 0)) + " variables")


# ## Ridge
# Create instance
ridgeCV = RidgeCV(alphas=[1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2,
                          1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100])
# Fit the model on the training set
model_ridge = ridgeCV.fit(X_train, y_train)
alpha = model_ridge.alpha_
print("Ridge best alpha :", alpha)
# Predict
y_pred_ridge_train = model_ridge.predict(X_train)
# Test
y_pred_ridge_test = model_ridge.predict(X_test)
# Results
print("Ridge Training set RMSE: : {:.4f}".format(rmse(y_train, y_pred_ridge_train)))
print("Ridge Test set RMSE: : {:.4f}".format(rmse(y_test, y_pred_ridge_test)))
# Add to the final comparison dictionary
model_score['ridge'] = rmse(y_test, y_pred_ridge_test)


# ## Random Forest
# Create instance
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True,
                           bootstrap=True, random_state=42)
# Fit the model on the training set
model_rf = rf.fit(X_train, y_train)
# Predict
y_pred_rf_train = model_rf.predict(X_train)
# Test
y_pred_rf_test = model_rf.predict(X_test)
# Results
print("Random Forest Training set RMSE: : {:.4f}".format(rmse(y_train, y_pred_rf_train)))
print("Random Forest Test set RMSE: : {:.4f}".format(rmse(y_test, y_pred_rf_test)))
# Add to the final comparison dictionary
model_score['rf'] = rmse(y_test, y_pred_rf_test)


# ## Gradient Boosting Regressor
# Create instance
gbr = GradientBoostingRegressor()
# Fit the model on the training set
model_gbr = gbr.fit(X_train, y_train)
# Predict
y_pred_gbr_train = model_gbr.predict(X_train)
# Test
y_pred_gbr_test = model_gbr.predict(X_test)
# Results
print("Gradient Boosting Training set RMSE: : {:.4f}".format(rmse(y_train, y_pred_gbr_train)))
print("Gradient Boosting Test set RMSE: : {:.4f}".format(rmse(y_test, y_pred_gbr_test)))
# Add to the final comparison dictionary
model_score['gbr'] = rmse(y_test, y_pred_gbr_test)


# ## Light Gradient Boosting Regressor
# Create instance
lightgbm = LGBMRegressor()
# Fit the model on the training set
model_lgbm = lightgbm.fit(X_train, y_train)
# Predict
y_pred_lgbm_train = model_lgbm.predict(X_train)
# Test
y_pred_lgbm_test = model_lgbm.predict(X_test)
# Results
print("Light Gradient Boosting Training set RMSE: : {:.4f}".format(rmse(y_train, y_pred_lgbm_train)))
print("Light Gradient Boosting Test set RMSE: : {:.4f}".format(rmse(y_test, y_pred_lgbm_test)))
# Add to the final comparison dictionary
model_score['lgbm'] = rmse(y_test, y_pred_lgbm_test)


# ## Extreme Gradient Boosting Regressor
# Create instance
xgb = xgboost.XGBRegressor(objective='reg:squarederror')
# Fit the model on the training set
model_xgb = xgb.fit(X_train, y_train)
# Predict
y_pred_xgb_train = model_xgb.predict(X_train)
# Test
y_pred_xgb_test = model_xgb.predict(X_test)
# Results
print("Extreme Gradient Boosting Training set RMSE: : {:.4f}".format(rmse(y_train, y_pred_xgb_train)))
print("Extreme Gradient Boosting Test set RMSE: : {:.4f}".format(rmse(y_test, y_pred_xgb_test)))
# Add to the final comparison dictionary
model_score['xgb'] = rmse(y_test, y_pred_xgb_test)


# ## AdaBoost
# Create instance
adab = AdaBoostRegressor()
# Fit the model on the training set
model_adab = adab.fit(X_train, y_train)
# Predict
y_pred_adb_train = model_adab.predict(X_train)
# Test
y_pred_adb_test = model_adab.predict(X_test)
# Results
print("AdaBoost Training set RMSE: : {:.4f}".format(rmse(y_train, y_pred_adb_train)))
print("AdaBoost Test set RMSE: : {:.4f}".format(rmse(y_test, y_pred_adb_test)))
# Add to the final comparison dictionary
model_score['adb'] = rmse(y_test, y_pred_adb_test)


model_score
# There is some overfitting in the model as it performs worse on the test set.
# But let’s say it is good enough and move forward to feature importance (measured on the training set performance).
# Some of the approaches can also be used for validation/OOB sets, to gain further interpretability on the unseen data.
