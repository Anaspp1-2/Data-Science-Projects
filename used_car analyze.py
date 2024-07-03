# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:50:50 2024

@author: anasp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


file_path = r"C:\Users\anasp\OneDrive\Pictures\personal\DATA SPARK\Kaggle datasets\used_cars_data.csv"
data = pd.read_csv(file_path)


columns_to_drop = ["S.No.", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "New_Price"]
data.drop(columns=columns_to_drop, axis=1, inplace=True)


data.nunique()
data.shape
data.describe().T
data.isna().sum()


from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
data["Price"] = pd.DataFrame(mean_imputer.fit_transform(data[["Price"]]))
print(data["Price"].isna().sum())


categorical_features = ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = one_hot_encoder.fit_transform(data[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(categorical_features))


df = pd.concat([data.drop(categorical_features, axis=1), encoded_df], axis=1)


df.columns = df.columns.astype(str)


plt.figure(figsize=(10, 6))
sns.histplot(data["Price"], bins=30, kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(16, 12))
sns.boxplot(x="Year", y="Price", data=data)
plt.title("Car Price by Year")
plt.xlabel('Year')
plt.ylabel("Price")
plt.show()

# Plot car price according to fuel type
plt.figure(figsize=(10, 6))
sns.boxplot(x="Fuel_Type", y="Price", data=data)
plt.title("Car Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x="Transmission", y="Price", data=data)
plt.title("Car Price by Transmission")
plt.xlabel("Transmission")
plt.ylabel("Price")
plt.show()


top_locations = data["Location"].value_counts().index[:10]
filtered_data = data[data["Location"].isin(top_locations)]
plt.figure(figsize=(16, 12))
sns.boxplot(x="Location", y="Price", data=filtered_data)
plt.title("Car Price by Location")
plt.xlabel("Location")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x="Owner_Type", y="Price", data=data)
plt.title("Car Price by Owner Type")
plt.xlabel("Owner Type")
plt.ylabel("Price")
plt.show()


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


clean_data = remove_outliers_iqr(data, 'Price')


plt.figure(figsize=(10, 6))
sns.boxplot(x=clean_data['Price'])
plt.title('Cleaned Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


def norm(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

df_norm = norm(df)


X = df_norm.drop(["Price"], axis=1)
y = df_norm["Price"]


X.columns = X.columns.astype(str)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)


y_pred = dt_reg.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Decision Tree Regressor Model")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

feature_importances = pd.DataFrame(dt_reg.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False) 
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
