#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

url = r"C:\Users\urvi9\Nashville_housing_data_2013_2016.csv"

df = pd.read_csv(url)
df


# In[98]:


print("Duplicates found:",df.duplicated().sum())

df = df.drop(columns = "Unnamed: 0.1")
df = df.drop(columns = "Unnamed: 0")
df = df.drop(columns = "Legal Reference")
df = df.drop(columns = "image")
df = df.drop(columns = "Parcel ID")

df = pd.get_dummies(df, columns=['Grade', 'Foundation Type'])
df


# In[99]:


column_number = 2
df.drop(df.columns[column_number], axis=1, inplace=True)

df.dropna(subset=['Acreage', 'Finished Area', 'Year Built', 'Bedrooms', 'Full Bath', 'Half Bath'], inplace=True)
df


# In[100]:


X = df[['Acreage', 'Finished Area', 'Year Built', 'Bedrooms', 'Full Bath', 'Half Bath']]
y = df['Sale Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

gradient_boost_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gradient_boost_model.fit(X_train, y_train)


# In[101]:


y_pred_lr = linear_reg_model.predict(X_test)
y_pred_dt = decision_tree_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_gb = gradient_boost_model.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_gb = mean_squared_error(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'MSE': [mse_lr, mse_dt, mse_rf, mse_gb],
    'MAE': [mae_lr, mae_dt, mae_rf, mae_gb],
    'R2': [r2_lr, r2_dt, r2_rf, r2_gb]
})

print("Performance Metrics for Each Model:")
print(metrics_df)


# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns
data_mse_mae = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'MSE': [mse_lr, mse_dt, mse_rf, mse_gb],
    'MAE': [mae_lr, mae_dt, mae_rf, mae_gb]
}
df_mse_mae = pd.DataFrame(data_mse_mae)
plt.figure(figsize=(8, 3.5))
sns.barplot(x='Model', y='MSE', data=df_mse_mae, color='purple', label='MSE')
plt.title('MSE for Each Model')
plt.xlabel('Model')
plt.legend()
plt.show()

plt.figure(figsize=(8, 3.5))
sns.barplot(x='Model', y='MAE', data=df_mse_mae, color='orange', label='MAE')
plt.title('MAE for Each Model')
plt.xlabel('Model')
plt.legend()
plt.show()


# In[ ]:




