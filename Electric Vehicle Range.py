#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


data=pd.read_csv(r'C:\Users\anasp\OneDrive\Pictures\personal\DATA SPARK\EXCEL\Electric_Vehicle_Population_Data.csv')


# In[3]:


data


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.nunique()


# In[9]:


data["Make"].value_counts().index


# In[10]:


data["Make"].value_counts().values


# In[11]:


data.head()


# In[12]:


data.tail()


# In[13]:


data.isna().sum()


# In[14]:


mode_imputer=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
data["Model"]=pd.DataFrame(mode_imputer.fit_transform(data[["Model"]]))
range_imputer = SimpleImputer(strategy="mean")
data["Electric Range"] = range_imputer.fit_transform(data[["Electric Range"]])


# In[15]:


data.drop(["VIN (1-10)","County","City","State","Postal Code","Legislative District","DOL Vehicle ID","Vehicle Location","Electric Utility","2020 Census Tract"],axis=1,inplace=True)


# In[16]:


plt.figure(figsize=(8, 6))
sns.histplot(data['Electric Range'], kde=True, bins=30)
plt.title("Distribution of Electric Range")
plt.show()


# In[17]:


IQR=data['Electric Range'].quantile(0.75)-data['Electric Range'].quantile(0.25)
lower_limit = data['Electric Range'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Electric Range'].quantile(0.75) + (IQR * 1.5)


# In[18]:


outliers_df = np.where(data['Electric Range'] > upper_limit, True, np.where(data['Electric Range'] < lower_limit, True, False))
df_trimmed = data.loc[~(outliers_df)]


# In[19]:


plt.figure(figsize=(8, 6))
sns.histplot(df_trimmed['Electric Range'], kde=True, bins=30)
plt.title("Distribution of Electric Range")
plt.show()


# In[20]:


plt.figure(figsize=(8, 6))
sns.violinplot(x="Electric Vehicle Type", y="Electric Range", data=df_trimmed)
plt.title("Electric Range Distribution by Electric Vehicle Type")
plt.show()


# In[21]:


sns.jointplot(x="Model Year", y="Electric Range", data=df_trimmed, kind="scatter", height=8)
plt.show()


# In[22]:


plt.figure(figsize=(12, 6))
sns.boxplot(x="Make", y="Electric Range", data=df_trimmed)
plt.title("Electric Range by Make")
plt.xticks(rotation=90)
plt.show()


# In[23]:


plt.figure(figsize=(8, 6))
sns.barplot(x="Electric Vehicle Type", y="Electric Range", data=df_trimmed, ci=None)
plt.title("Average Electric Range by Electric Vehicle Type")
plt.show()


# In[24]:


df_zero_range = df_trimmed[df_trimmed['Electric Range'] == 0]
df_non_zero_range = df_trimmed[df_trimmed['Electric Range'] > 0]


# In[25]:


print(f"Vehicles with Zero Electric Range: {df_zero_range.shape[0]}")
print(f"Vehicles with Non-Zero Electric Range: {df_non_zero_range.shape[0]}")


# In[26]:


df_non_zero_range['Log_Electric_Range'] = np.log(df_non_zero_range['Electric Range'])


# In[27]:


plt.figure(figsize=(8, 6))
sns.histplot(df_non_zero_range['Log_Electric_Range'], kde=True, bins=30)
plt.title("Distribution of Log-Transformed Electric Range")
plt.show()


# In[28]:


X_non_zero = df_non_zero_range.drop(['Electric Range', 'Log_Electric_Range'], axis=1)
y_non_zero = df_non_zero_range['Log_Electric_Range']
X_train_non_zero, X_test_non_zero, y_train_non_zero, y_test_non_zero = train_test_split(X_non_zero, y_non_zero, test_size=0.2, random_state=42)


# In[29]:


def create_custom_pipeline(categorical_cols, numerical_cols, scaling_method=None):
    # Define transformers for categorical and numerical columns
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore'))])

    if scaling_method:
        if scaling_method == 'standard':
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        elif scaling_method == 'robust':
            numerical_transformer = Pipeline(steps=[('scaler', RobustScaler())])
        elif scaling_method == 'minmax':
            numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    else:
        numerical_transformer = "passthrough"

    # Specify which columns to apply each transformer to
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Include the preprocessor in your main pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline


# In[30]:


categorical_cols = ["Make",'Model','Electric Vehicle Type','Clean Alternative Fuel Vehicle (CAFV) Eligibility']
numerical_cols = ['Model Year']

# Applying the pipeine to categorical and numerical columns if required
pipeline = create_custom_pipeline(categorical_cols, numerical_cols,scaling_method='standard')


# Fit and transform the data
X_preprocessed_train_non_zero = pipeline.fit_transform(X_train_non_zero)
X_preprocessed_test_non_zero = pipeline.transform(X_test_non_zero)


# In[31]:


model = LinearRegression()
model.fit(X_preprocessed_train_non_zero, y_train_non_zero)

# Predict the log-transformed values
y_pred_log = model.predict(X_preprocessed_test_non_zero)

# Inverse log transformation to get back to original scale
y_pred = np.exp(y_pred_log)

rmse = np.sqrt(mean_squared_error(np.exp(y_test_non_zero), y_pred))
print(f"Root Mean Squared Error after Log Transformation: {rmse:.2f}")


# In[32]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[33]:


rf_model = RandomForestRegressor(random_state=42)

rf_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5]
}


# In[34]:


rf_random_search = RandomizedSearchCV(rf_model, rf_param_grid, n_iter=5, cv=3, random_state=42, scoring='neg_mean_squared_error', n_jobs=-1
)

rf_random_search.fit(X_preprocessed_train_non_zero, y_train_non_zero)

best_rf_model = rf_random_search.best_estimator_

# Predict on the test set
y_pred_rf_log = best_rf_model.predict(X_preprocessed_test_non_zero)

# Inverse log transformation to get back to original scale
y_pred_rf = np.exp(y_pred_rf_log)

# Calculate RMSE for Random Forest
rf_rmse = np.sqrt(mean_squared_error(np.exp(y_test_non_zero), y_pred_rf))
print(f"Random Forest RMSE: {rf_rmse:.2f}")


# In[35]:


feature_importances = best_rf_model.feature_importances_
features = pipeline.get_feature_names_out()
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




