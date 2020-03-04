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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Import the data after feature engineering
df = pd.read_csv('https://raw.githubusercontent.com/jeremyndeby/Toulouse_Apt_Rental_Price/master/EDA_feature_engineering/data_for_modeling.csv')

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


# Define a rmse evaluation function
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# Create a results function
def results_func(model_name, y_pred_model_train, y_pred_model_test):
    # Create metrics
    model_r2_train = r2_score(y_train, y_pred_model_train)
    model_r2_test = r2_score(y_test, y_pred_model_test)
    model_mae_train = mean_absolute_error(y_train, y_pred_model_train)
    model_mae_test = mean_absolute_error(y_test, y_pred_model_test)
    model_rmse_train = rmse(y_train, y_pred_model_train)
    model_rmse_test = rmse(y_test, y_pred_model_test)

    # Print results
    print(model_name + "\n\tTraining set R^2: : {:.4f}".format(model_r2_train))
    print("\tTest set R^2: : {:.4f}".format(model_r2_test))
    print("\tTraining set MAE: : {:.4f}".format(model_mae_train))
    print("\tTest set MAE: : {:.4f}".format(model_mae_test))
    print("\tTraining set RMSE: : {:.4f}".format(model_rmse_train))
    print("\tTest set RMSE: : {:.4f}\n".format(model_rmse_test))
    return model_r2_train, model_r2_test, model_mae_train, model_mae_test, model_rmse_train, model_rmse_test


# Create a feature importances plot function
def plot_feat_importances_func(model_name, model):
    # Feature importances
    ft_weights = pd.DataFrame(model.feature_importances_, columns=['weight'], index=X_train.columns)
    ft_weights.sort_values('weight', ascending=False, inplace=True)
    # ft_weights.head(20)

    # Plotting feature importances
    plt.figure(figsize=(10, 25))
    plt.barh(ft_weights.index, ft_weights.weight, align='center')
    plt.title(model_name + " Feature importances", fontsize=14)
    plt.xlabel("Feature importance")
    plt.margins(y=0.01)
    plt.show()


# ### Create a function that plots results
def plot_results_func(results_table):
    plt.rcParams["axes.labelsize"] = 18
    plt.figure(figsize=(26, 18))
    sns.set_palette("PuBuGn_d")

    plt.subplot(3, 1, 1)
    plt.title("Model Comparison in terms of R-squared, MAE and RMSE", fontsize=20, weight='bold')
    g1 = sns.barplot(x="Model", y="R-squared", data=results_table, palette="deep")
    plt.ylim(0.4, 0.9)
    g1.set_xlabel("")
    g1.tick_params(labelsize=16)
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x() + p.get_width() / 2., height + 0.025, "{:1.4f}".format(height), ha="center", fontsize=16,
                weight='bold')

    plt.subplot(3, 1, 2)
    g2 = sns.barplot(x="Model", y="MAE", data=results_table, palette="deep")
    plt.ylim(0.05, 0.15)
    g2.set_xlabel("")
    g2.tick_params(labelsize=16)
    for p in g2.patches:
        height = p.get_height()
        g2.text(p.get_x() + p.get_width() / 2., height + 0.005, "{:1.4f}".format(height), ha="center", fontsize=16,
                weight='bold')

    plt.subplot(3, 1, 3)
    g3 = sns.barplot(x="Model", y="RMSE", data=results_table, palette="deep")
    plt.ylim(0.1, 0.2)
    g3.set_xlabel("")
    g3.tick_params(labelsize=16)
    for p in g3.patches:
        height = p.get_height()
        g3.text(p.get_x() + p.get_width() / 2., height + 0.005, "{:1.4f}".format(height), ha="center", fontsize=16,
                weight='bold')

    plt.savefig('model_comparison.png')
    plt.show()


# ### Compare models

'''LINEAR REGRESSION'''
# Sometimes, simple models, outperformed more complex models like Random Forest and xgboost,
# especially on small datasets.
# Create instance
lr = linear_model.LinearRegression()
# Fit the model on the training set
lr.fit(X_train, y_train)
# Predict
y_pred_lr_train = lr.predict(X_train)
# Test
y_pred_lr_test = lr.predict(X_test)

# Results
lr_r2_train, lr_r2_test, lr_mae_train, lr_mae_test, lr_rmse_train, lr_rmse_test = results_func('Linear Regression:',
                                                                                               y_pred_lr_train,
                                                                                               y_pred_lr_test)

'''LASSO REGRESSION'''
# Compute the cross-validation score with default hyper-parameters
# Create instance
lassoCV = LassoCV(alphas=[1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2,
                          1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100])
# Fit the model on the training set
model_lasso = lassoCV.fit(X_train, y_train)
alpha_l = model_lasso.alpha_
# Predict
y_pred_lasso_train = model_lasso.predict(X_train)
# Test
y_pred_lasso_test = model_lasso.predict(X_test)

coef = pd.Series(model_lasso.coef_, index=X_train.columns)

# Results
lasso_r2_train, lasso_r2_test, lasso_mae_train, lasso_mae_test, lasso_rmse_train, lasso_rmse_test = results_func(
    'Lasso:', y_pred_lasso_train, y_pred_lasso_test)
print("Lasso best alpha :", alpha_l)
print("\nLasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
    sum(coef == 0)) + " variables")

'''RIDGE REGRESSION'''
# Compute the cross-validation score with default hyper-parameters
# Create instance
ridgeCV = RidgeCV(alphas=[1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2,
                          1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100])
# Fit the model on the training set
model_ridge = ridgeCV.fit(X_train, y_train)
alpha = model_ridge.alpha_
# Predict
y_pred_ridge_train = model_ridge.predict(X_train)
# Test
y_pred_ridge_test = model_ridge.predict(X_test)
# Results
ridge_r2_train, ridge_r2_test, ridge_mae_train, ridge_mae_test, ridge_rmse_train, ridge_rmse_test = results_func(
    'Ridge:', y_pred_ridge_train, y_pred_ridge_test)
print("Ridge best alpha :", alpha)

'''RANDOM FOREST REGRESSOR (tuned using RandomizedSearchCV and GridSearchCV)'''
# Create instance
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=8,
                           max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=2, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=1500,
                           n_jobs=-1, oob_score=False, random_state=42,
                           verbose=0, warm_start=False)
# Fit the model on the training set
rf.fit(X_train, y_train)
# Predict
y_pred_rf_train = rf.predict(X_train)
# Test
y_pred_rf_test = rf.predict(X_test)
# Results
rf_r2_train, rf_r2_test, rf_mae_train, rf_mae_test, rf_rmse_train, rf_rmse_test = results_func('Random Forest:',
                                                                                               y_pred_rf_train,
                                                                                               y_pred_rf_test)
# Feature importances
plot_feat_importances_func('Random Forest', rf)

'''GRADIENT BOOSTING REGRESSOR (default parameters)'''
# Create instance
gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=200,
                                max_depth=7,
                                min_samples_split=60, min_samples_leaf=11,
                                max_features=25, subsample=0.75,
                                random_state=42)

# Fit the model on the training set
gbr.fit(X_train, y_train)
# Predict
y_pred_gbr_train = gbr.predict(X_train)
# Test
y_pred_gbr_test = gbr.predict(X_test)
# Results
gbr_r2_train, gbr_r2_test, gbr_mae_train, gbr_mae_test, gbr_rmse_train, gbr_rmse_test = results_func('Gradient '
                                                                                                     'Boosting:',
                                                                                                     y_pred_gbr_train,
                                                                                                     y_pred_gbr_test)
# Feature importances
plot_feat_importances_func('Gradient Boosting', gbr)

'''EXTREME GRADIENT BOOSTING REGRESSOR (tuned using GridSearchCV)'''
# Create instance mae
xgbreg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42,
                          max_depth=8, min_child_weight=2,
                          subsample=.95, colsample_bytree=.85,
                          reg_lambda=1, reg_alpha=0.1,
                          eta=.01, n_jobs=-1)

# Extreme Gradient Boosting Regressor:
#  	Extreme Gradient Boosting Training set R^2: : 0.9642
# 	Extreme Gradient Boosting Test set R^2: : 0.7692
# 	Extreme Gradient Boosting Training set RMSE: : 0.0588
# 	Extreme Gradient Boosting Test set RMSE: : 0.1513
# 	Extreme Gradient Boosting Training set MAE: : 0.0365
# 	Extreme Gradient Boosting Test set MAE: : 0.0905

xgbreg_v2 = xgb.XGBRegressor(objective='reg:squarederror', random_state=42,
                             max_depth=4, min_child_weight=4,
                             subsample=.75, colsample_bytree=.95,
                             reg_lambda=5, reg_alpha=1e-05,
                             eta=.01)

# Extreme Gradient Boosting Regressor:
#  	Extreme Gradient Boosting Training set R^2: : 0.8677
# 	Extreme Gradient Boosting Test set R^2: : 0.7637
# 	Extreme Gradient Boosting Training set RMSE: : 0.1131
# 	Extreme Gradient Boosting Test set RMSE: : 0.1530
# 	Extreme Gradient Boosting Training set MAE: : 0.0753
# 	Extreme Gradient Boosting Test set MAE: : 0.0928

# Fit the model on the training set
xgbreg.fit(X_train, y_train)
# Predict
y_pred_xgb_train = xgbreg.predict(X_train)
# Test
y_pred_xgb_test = xgbreg.predict(X_test)
# Results
xgb_r2_train, xgb_r2_test, xgb_mae_train, xgb_mae_test, xgb_rmse_train, xgb_rmse_test = results_func('Extreme '
                                                                                                     'Gradient '
                                                                                                     'Boosting:',
                                                                                                     y_pred_xgb_train,
                                                                                                     y_pred_xgb_test)
# Feature importances
plot_feat_importances_func('Extreme Gradient Boosting', xgbreg)

'''LIGHT GRADIENT BOOSTING REGRESSOR (default parameters)'''
# Create instance
lgbm = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
                     importance_type='split', learning_rate=0.01, max_depth=4,
                     metric='mse', min_child_samples=10, min_child_weight=2,
                     min_split_gain=0.0, n_estimators=1800, n_jobs=-1, num_leaves=100,
                     objective='regression', random_state=42, reg_alpha=0,
                     reg_lambda=20, silent=True, subsample=0.75,
                     subsample_for_bin=200000, subsample_freq=0)
# Fit the model on the training set
lgbm.fit(X_train, y_train)
# Predict
y_pred_lgbm_train = lgbm.predict(X_train)
# Test
y_pred_lgbm_test = lgbm.predict(X_test)
# Results
lgbm_r2_train, lgbm_r2_test, lgbm_mae_train, lgbm_mae_test, lgbm_rmse_train, lgbm_rmse_test = results_func('Light '
                                                                                                           'Gradient '
                                                                                                           'Boosting:',
                                                                                                           y_pred_lgbm_train,
                                                                                                           y_pred_lgbm_test)
# Feature importances
plot_feat_importances_func('Light Gradient Boosting', lgbm)

# ## Model Evaluation
# In this section, we will put together the results from all four models
# and compare them side by side in order to evaluate their performance.
# The metrics that we use for evaluation are R-squared, Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

# combine all models' results into one dataframe
data_models = {'Model': ['Linear Regression', 'Lasso', 'Ridge', 'Random Forest', 'Gradient Boosting',
                         'Extreme Gradient Boosting', 'Light Gradient Boosting'],
               'R-squared': [lr_r2_test, lasso_r2_test, ridge_r2_test, rf_r2_test,
                             gbr_r2_test, xgb_r2_test, lgbm_r2_test],
               'MAE': [lr_mae_test, lasso_mae_test, ridge_mae_test, rf_mae_test,
                       gbr_mae_test, xgb_mae_test, lgbm_mae_test],
               'RMSE': [lr_rmse_test, lasso_rmse_test, ridge_rmse_test, rf_rmse_test,
                        gbr_rmse_test, xgb_rmse_test, lgbm_rmse_test]}

results = pd.DataFrame(data=data_models)

plot_results_func(results)


