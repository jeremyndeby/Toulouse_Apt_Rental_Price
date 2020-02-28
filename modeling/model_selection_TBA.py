#!/usr/bin/env python
# coding: utf-8

# ### Model Selection ### #

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Import the data after feature engineering
df = pd.read_csv(
    'https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/EDA_feature_engineering/data_for_modeling.csv')


# ### Splitting data

# Set the target variable
y = df.rent.values

# Set predictors
df.drop(['rent'], axis=1, inplace=True)
X = df

# Split data (random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

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

'''Baseline'''
# "Learn" the mean from the training data
model_mean = np.mean(y_train)
# Get predictions on the test set
y_pred_mean_train = np.ones(y_train.shape) * model_mean
y_pred_mean_test = np.ones(y_test.shape) * model_mean


mean_train = np.mean(y_train)
# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train
# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)

# Results
mean_rmse_train = rmse(y_train, y_pred_mean_train)
mean_rmse_test = rmse(y_test, y_pred_mean_test)
print("Baseline predictor:\n \tMean Training set RMSE: : {:.4f}".format(mean_rmse_train))
print("\tMean Test set RMSE: : {:.4f}".format(mean_rmse_test))
mean_mae_train = mean_absolute_error(y_train, y_pred_mean_train)
mean_mae_test = mean_absolute_error(y_test, y_pred_mean_test)
print("\tMean Training set MAE: : {:.4f}".format(mean_mae_train))
print("\tMean Test set MAE: : {:.4f}\n".format(mean_mae_test))


'''Linear Regression'''
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
lr_r2_train = r2_score(y_train, y_pred_lr_train)
lr_r2_test = r2_score(y_test, y_pred_lr_test)
print("Linear Regression:\n \tLinearRegression Training set R^2: : {:.4f}".format(lr_r2_train))
print("\tLinearRegression Test set R^2: : {:.4f}".format(lr_r2_test))
lr_rmse_train = rmse(y_train, y_pred_lr_train)
lr_rmse_test = rmse(y_test, y_pred_lr_test)
print("\tLinearRegression Training set RMSE: : {:.4f}".format(lr_rmse_train))
print("\tLinearRegression Test set RMSE: : {:.4f}".format(lr_rmse_test))
lr_mae_train = mean_absolute_error(y_train, y_pred_lr_train)
lr_mae_test = mean_absolute_error(y_test, y_pred_lr_test)
print("\tLinearRegression Training set MAE: : {:.4f}".format(lr_mae_train))
print("\tLinearRegression Test set MAE: : {:.4f}\n".format(lr_mae_train))


'''Lasso Regression'''
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
lasso_r2_train = r2_score(y_train, y_pred_lasso_train)
lasso_r2_test = r2_score(y_test, y_pred_lasso_test)
print("Lasso Regression:\n \tLasso Training set R^2: : {:.4f}".format(lasso_r2_train))
print("\tLasso Test set R^2: : {:.4f}".format(lasso_r2_test))
lasso_rmse_train = rmse(y_train, y_pred_lasso_train)
lasso_rmse_test = rmse(y_test, y_pred_lasso_test)
print("\tLasso Training set RMSE: : {:.4f}".format(lasso_rmse_train))
print("\tLasso Test set RMSE: : {:.4f}".format(lasso_rmse_test))
lasso_mae_train = mean_absolute_error(y_train, y_pred_lasso_train)
lasso_mae_test = mean_absolute_error(y_test, y_pred_lasso_test)
print("\tLasso Training set MAE: : {:.4f}".format(lasso_mae_train))
print("\tLasso Test set MAE: : {:.4f}".format(lasso_mae_train))
coef = pd.Series(model_lasso.coef_, index=X_train.columns)
print("\nLasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
    sum(coef == 0)) + " variables")


'''Ridge Regression'''
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
ridge_r2_train = r2_score(y_train, y_pred_ridge_train)
ridge_r2_test = r2_score(y_test, y_pred_ridge_test)
print("Ridge Regression:\n \tRidge Training set R^2: : {:.4f}".format(ridge_r2_train))
print("\tRidge Test set R^2: : {:.4f}".format(ridge_r2_test))
ridge_rmse_train = rmse(y_train, y_pred_ridge_train)
ridge_rmse_test = rmse(y_test, y_pred_ridge_test)
print("\tRidge Training set RMSE: : {:.4f}".format(ridge_rmse_train))
print("\tRidge Test set RMSE: : {:.4f}".format(ridge_rmse_test))
ridge_mae_train = mean_absolute_error(y_train, y_pred_ridge_train)
ridge_mae_test = mean_absolute_error(y_test, y_pred_ridge_test)
print("\tRidge Training set MAE: : {:.4f}".format(ridge_mae_train))
print("\tRidge Test set MAE: : {:.4f}\n".format(ridge_mae_train))


'''Random Forest Regressor (default parameters)'''
# Grid search is a tuning technique that attempts to compute the optimum values of hyperparameters.
# It can be extremely helpful to help us determine the optimal values in an elegant manner.
# Create instance
rf = RandomForestRegressor(random_state=42)
# Fit the model on the training set
model_rf = rf.fit(X_train, y_train)
# Predict
y_pred_rf_train = model_rf.predict(X_train)
# Test
y_pred_rf_test = model_rf.predict(X_test)
# Results
rf_r2_train = r2_score(y_train, y_pred_rf_train)
rf_r2_test = r2_score(y_test, y_pred_rf_test)
print("Random Forest Regressor:\n \tRandom Forest Training set R^2: : {:.4f}".format(rf_r2_train))
print("\tRandom Forest Test set R^2: : {:.4f}".format(rf_r2_test))
rf_rmse_train = rmse(y_train, y_pred_rf_train)
rf_rmse_test = rmse(y_test, y_pred_rf_test)
print("\tRandom Forest Training set RMSE: : {:.4f}".format(rf_rmse_train))
print("\tRandom Forest Test set RMSE: : {:.4f}".format(rf_rmse_test))
rf_mae_train = mean_absolute_error(y_train, y_pred_rf_train)
rf_mae_test = mean_absolute_error(y_test, y_pred_rf_test)
print("\tRandom Forest Training set MAE: : {:.4f}".format(rf_mae_train))
print("\tRandom Forest Test set MAE: : {:.4f}\n".format(rf_mae_test))


'''Gradient Boosting Regressor (default parameters)'''
# Create instance
gbr = GradientBoostingRegressor(random_state=42)
# Fit the model on the training set
model_gbr = gbr.fit(X_train, y_train)
# Predict
y_pred_gbr_train = model_gbr.predict(X_train)
# Test
y_pred_gbr_test = model_gbr.predict(X_test)
# Results
gbr_r2_train = r2_score(y_train, y_pred_gbr_train)
gbr_r2_test = r2_score(y_test, y_pred_gbr_test)
print("Gradient Boosting Regressor:\n \tGradient Boosting Training set R^2: : {:.4f}".format(gbr_r2_train))
print("\tGradient Boosting Test set R^2: : {:.4f}".format(gbr_r2_test))
gbr_rmse_train = rmse(y_train, y_pred_gbr_train)
gbr_rmse_test = rmse(y_test, y_pred_gbr_test)
print("\tGradient Boosting Training set RMSE: : {:.4f}".format(gbr_rmse_train))
print("\tGradient Boosting Test set RMSE: : {:.4f}".format(gbr_rmse_test))
gbr_mae_train = mean_absolute_error(y_train, y_pred_gbr_train)
gbr_mae_test = mean_absolute_error(y_test, y_pred_gbr_test)
print("\tGradient Boosting Training set MAE: : {:.4f}".format(gbr_mae_train))
print("\tGradient Boosting Test set MAE: : {:.4f}\n".format(gbr_mae_test))


'''Extreme Gradient Boosting Regressor (tuned )'''
# Create instance
xgbreg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42,
                          max_depth=7, min_child_weight=1,
                           reg_lambda=1, reg_alpha=0,
                           subsample=.85, colsample_bytree=.9,
                          eta=.01)
# Fit the model on the training set
model_xgb = xgbreg.fit(X_train, y_train)
# Predict
y_pred_xgb_train = model_xgb.predict(X_train)
# Test
y_pred_xgb_test = model_xgb.predict(X_test)
# Results
xgb_r2_train = r2_score(y_train, y_pred_xgb_train)
xgb_r2_test = r2_score(y_test, y_pred_xgb_test)
print("Extreme Gradient Boosting Regressor:\n \tExtreme Gradient Boosting Training set R^2: : {:.4f}".format(xgb_r2_train))
print("\tExtreme Gradient Boosting Test set R^2: : {:.4f}".format(xgb_r2_test))
xgb_rmse_train = rmse(y_train, y_pred_xgb_train)
xgb_rmse_test = rmse(y_test, y_pred_xgb_test)
print("\tExtreme Gradient Boosting Training set RMSE: : {:.4f}".format(xgb_rmse_train))
print("\tExtreme Gradient Boosting Test set RMSE: : {:.4f}".format(xgb_rmse_test))
xgb_mae_train = mean_absolute_error(y_train, y_pred_xgb_train)
xgb_mae_test = mean_absolute_error(y_test, y_pred_xgb_test)
print("\tExtreme Gradient Boosting Training set MAE: : {:.4f}".format(xgb_mae_train))
print("\tExtreme Gradient Boosting Test set MAE: : {:.4f}\n".format(xgb_mae_test))


'''Light Gradient Boosting Regressor (default parameters)'''
# Create instance
lightgbm = LGBMRegressor(random_state=42)
# Fit the model on the training set
model_lgbm = lightgbm.fit(X_train, y_train)
# Predict
y_pred_lgbm_train = model_lgbm.predict(X_train)
# Test
y_pred_lgbm_test = model_lgbm.predict(X_test)
# Results
lgbm_r2_train = r2_score(y_train, y_pred_lgbm_train)
lgbm_r2_test = r2_score(y_test, y_pred_lgbm_test)
print("Light Gradient Boosting Regressor:\n \tLight Gradient Boosting Training set R^2: : {:.4f}".format(lgbm_r2_train))
print("\tLight Gradient Boosting Test set R^2: : {:.4f}".format(lgbm_r2_test))
lgbm_rmse_train = rmse(y_train, y_pred_lgbm_train)
lgbm_rmse_test = rmse(y_test, y_pred_lgbm_test)
print("\tLight Gradient Boosting Training set RMSE: : {:.4f}".format(lgbm_rmse_train))
print("\tLight Gradient Boosting Test set RMSE: : {:.4f}".format(lgbm_rmse_test))
lgbm_mae_train = mean_absolute_error(y_train, y_pred_lgbm_train)
lgbm_mae_test = mean_absolute_error(y_test, y_pred_lgbm_test)
print("\tLight Gradient Boosting Training set MAE: : {:.4f}".format(lgbm_mae_train))
print("\tLight Gradient Boosting Test set MAE: : {:.4f}\n".format(lgbm_mae_test))



'''Adaptive Boosting Regressor (default parameters)'''
# Create instance
adab = AdaBoostRegressor(random_state=42)
# Fit the model on the training set
model_adab = adab.fit(X_train, y_train)
# Predict
y_pred_adb_train = model_adab.predict(X_train)
# Test
y_pred_adb_test = model_adab.predict(X_test)
adb_r2_train = r2_score(y_train, y_pred_adb_train)
adb_r2_test = r2_score(y_test, y_pred_adb_test)
print("Adaptive Boosting Regressor:\n \tAdaptive Boosting Training set R^2: : {:.4f}".format(adb_r2_train))
print("\tAdaptive Boosting Test set R^2: : {:.4f}".format(adb_r2_test))
adb_rmse_train = rmse(y_train, y_pred_adb_train)
adb_rmse_test = rmse(y_test, y_pred_adb_test)
print("\tAdaptive Boosting Training set RMSE: : {:.4f}".format(adb_rmse_train))
print("\tAdaptive Boosting Test set RMSE: : {:.4f}".format(adb_rmse_test))
adb_mae_train = mean_absolute_error(y_train, y_pred_adb_train)
adb_mae_test = mean_absolute_error(y_test, y_pred_adb_test)
print("\tAdaptive Boosting Training set MAE: : {:.4f}".format(adb_mae_train))
print("\tAdaptive Boosting Test set MAE: : {:.4f}\n".format(adb_mae_test))


# There is some overfitting in the model as it performs worse on the test set.
# But let’s say it is good enough and move forward to feature importance (measured on the training set performance).
# Some of the approaches can also be used for validation/OOB sets, to gain further interpretability on the unseen data.


# ## Model Evaluation
# In this section, we will put together the results from all four models
# and compare them side by side in order to evaluate their performance.
# The metrics that we use for evaluation are R-squared, Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

# combine all models' results into one dataframe
data_models = {'Model': ['Linear Regression', 'Lasso', 'Ridge', 'Random Forest', 'Gradient Boosting',
                         'Light Gradient Boosting', 'Extreme Gradient Boosting', 'Adaptive Boosting'],
               'R-squared': [lr_r2_test, lasso_r2_test, ridge_r2_test, rf_r2_test,
                             gbr_r2_test, lgbm_r2_test, xgb_r2_test, adb_r2_test],
               'RMSE': [lr_rmse_test, lasso_rmse_test, ridge_rmse_test, rf_rmse_test,
                        gbr_rmse_test, lgbm_rmse_test, xgb_rmse_test, adb_rmse_test],
               'MAE': [lr_mae_test, lasso_mae_test, ridge_mae_test, rf_mae_test,
                       gbr_mae_test, lgbm_mae_test, xgb_mae_test, adb_mae_test]}

results = pd.DataFrame(data=data_models)
results


# visualize the results using bar charts
# sns.set(rc={"figure.figsize":(10, 12)})

plt.rcParams["axes.labelsize"] = 18
plt.figure(figsize=(26, 18))
sns.set_palette("PuBuGn_d")

plt.subplot(3, 1, 1)
plt.title("Model Comparison in terms of R-squared, RMSE and MAE", fontsize=20, weight='bold')
g1 = sns.barplot(x="Model", y="R-squared", data=results, palette="deep")
plt.ylim(0.4, 0.9)
g1.set_xlabel("")
g1.tick_params(labelsize=16)
for p in g1.patches:
    height = p.get_height()
    g1.text(p.get_x()+p.get_width()/2., height+0.025, "{:1.4f}".format(height), ha="center", fontsize=16, weight='bold')

plt.subplot(3, 1, 2)
g2 = sns.barplot(x="Model", y="RMSE", data=results, palette="deep")
plt.ylim(0.1, 0.2)
g2.set_xlabel("")
g2.tick_params(labelsize=16)
for p in g2.patches:
    height = p.get_height()
    g2.text(p.get_x()+p.get_width()/2., height+0.005, "{:1.4f}".format(height), ha="center", fontsize=16, weight='bold')

plt.subplot(3, 1, 3)
g3 = sns.barplot(x="Model", y="MAE", data=results, palette="deep")
plt.ylim(0.05, 0.15)
g3.set_xlabel("")
g3.tick_params(labelsize=16)
for p in g3.patches:
    height = p.get_height()
    g3.text(p.get_x()+p.get_width()/2., height+0.005, "{:1.4f}".format(height), ha="center", fontsize=16, weight='bold')
plt.show()