#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


demand= pd.read_csv(r"D:\Data Science Project\US_Property\archive (4)\demand.csv")
supply= pd.read_csv(r"D:\Data Science Project\US_Property\archive (4)\supply.csv")


# In[3]:


demand.head()


# In[4]:


supply


# In[5]:


print("demand _shape",demand.shape)
print("Supply_shape",supply.shape)


# In[ ]:





# In[6]:


data=demand.merge(supply, on="DATE")
data


# In[7]:


data.info()


# In[8]:


data["DATE"]= pd.to_datetime(data["DATE"])
data.info()


# In[9]:


data.describe()


# In[10]:


data.columns


# In[21]:


data['Home_price_index'] = data['Home_price_index'].astype(float).round(1)
data['Montly_Supply']=data['Montly_Supply'].astype(float).round(1)
data['Permit']=data['Permit'].astype(float).round(1)
data['Total_consumption spending']=data['Total_consumption spending'].astype(float).round(1)
data['Housing_inventory']=data['Housing_inventory'].astype(float).round(1)
data['30y_mortgage']=data['30y_mortgage'].astype(float).round(1)
data['Consumer_sentiment']=data['Consumer_sentiment'].astype(float).round(1)
data['Interest_rates']=data['Interest_rates'].astype(float).round(1)
data['Median_sales_price']=data['Median_sales_price'].astype(float).round(1)
data['GDP']=data['GDP'].astype(float).round(1)


data.head()


# In[12]:


data.drop(columns='Home_Price_index', inplace=True)
data


# In[14]:


data.isna().sum()


# In[22]:


data.describe()


# In[32]:


data[data['Home_price_index'].isnull()]


# In[37]:


NA_index=data[data['Home_price_index'].isnull()].index
NA_index


# ### Droping The Null Value Column 

# In[43]:


data.dropna(inplace=True)
data


# ## Data Visualization

# In[45]:


sns.lineplot(data=data,x='DATE',y='Home_price_index',color='blue').set(title='Home prices index over the years')

# this plot shows the Home prices index over the last 20 years


# In[65]:


sns.lineplot(data=data,x='DATE',y='Montly_Supply',color='red').set(title='Monthly supply over the years')

# this plot shows the monthly supply over the last 20 years


# In[66]:


sns.lineplot(data=data,x='DATE',y='Permit',color='green').set(title='Permit over the years')

# this plot shows the Permit over the last 20 years


# In[67]:


sns.lineplot(data=data,x='DATE',y='Total_consumption spending',color='skyblue').set(title='Total_consumption spending over the years')


# In[68]:


#ploting the data in single fig using pyplot
figure,ax = plt.subplots(nrows=1,ncols=3,figsize=(18,5))

sns.lineplot(ax = ax[0],data =data,x='DATE',y='Housing_inventory').set(title='Housing_inventory over the last 20 years')
sns.lineplot(ax = ax[1],data=data,x='DATE',y='30y_mortgage').set(title='30y_mortgage over the last 20 years')
sns.lineplot(ax = ax[2],data=data,x='DATE',y='Consumer_sentiment').set(title='Consumer_sentiment over last 20 years')


# In[69]:


#ploting the data in single fig using pyplot
figure,ax = plt.subplots(nrows=1,ncols=3,figsize=(18,5))

sns.lineplot(ax = ax[0],data=data,x='DATE',y='Interest_rates').set(title='Interest_rates over the last 20 years')
sns.lineplot(ax = ax[1],data=data,x='DATE',y='Median_sales_price').set(title='Median_sales_price over the last 20 years')
sns.lineplot(ax = ax[2],data=data,x='DATE',y='GDP').set(title='GDP over the last 20 years')


# In[70]:


data.corr()


# In[72]:


plt.figure(figsize=(15,10))
sns.heatmap(data=data.corr(),annot=True).set(title='Correlation between factors impacted the home price')


# In[74]:


figure,ax = plt.subplots(nrows=1,ncols=3,figsize=(20,5))
sns.lineplot(ax = ax[0],data=data,x='DATE',y='Interest_rates')
sns.lineplot(ax = ax[0],data=data,x='DATE',y='30y_mortgage').set(title='Relation between Intrest rates and 30y_mortgage for 20 Y')

sns.lineplot(ax=ax[1],x = 'DATE',y='Montly_Supply',data =data)
sns.lineplot(ax=ax[1],x = 'DATE',y='Interest_rates',data =data).set(title='Relation between monthly supply and interest rates')

sns.lineplot(ax=ax[2],x = 'DATE',y='Consumer_sentiment',data =data)
sns.lineplot(ax=ax[2],x = 'DATE',y='Home_price_index',data =data).set(title='Relationship between consumer sentiment and home price index')


# ## MACHINE LEARNING MODEL

# In[110]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score


# In[102]:


data_wd=data.drop('DATE', axis=1)
data_wd
feature_wd = data_wd.drop('Home_price_index', axis=1)
target_wd = data_wd[['Home_price_index']]


# In[103]:


print(feature_wd.shape)
print(target_wd.shape)


# In[101]:


feature_wd = data.drop('Home_price_index', axis=1)
target_wd = data[['Home_price_index']]


# In[93]:


print(feature.shape)
print(target.shape)


# In[104]:


X_train, X_test, Y_train, Y_test = train_test_split(feature_wd, target_wd, random_state=10)


# In[105]:


print("The Shape X_Train",X_train.shape)
print("The Shape Y_Train",Y_train.shape)
print("The Shape X_Test",X_test.shape)
print("The Shape Y_Test",Y_test.shape)


# ## Linear Regression

# In[106]:


my_linear=LinearRegression()


# In[107]:


my_log_model = my_linear.fit(X_train, Y_train)


# In[116]:


my_log_reg_train = my_log_model.predict(X_train)
my_log_reg_test = my_log_model.predict(X_test)
r2_score(Y_test,my_log_reg_test)


# ## Random Forest Regressor

# In[115]:


regressor = RandomForestRegressor(n_estimators=100,max_depth=5)
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)

r2_score(Y_test,y_pred)


# In[118]:


k_range = range(1, 25)
knn_score = []

caler = StandardScaler()
X_train_scaled =  scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, Y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_score.append(r2_score(Y_test, knn_pred))

plt.plot(k_range,knn_score);
knn_score


# ## Ridge  Regression

# In[119]:



ridge_reg = Ridge(alpha=10)
ridge_reg.fit(X_train_scaled,Y_train)
y_pred_ridge= ridge_reg.predict(X_test_scaled)

ridge_reg.score(X_test_scaled,Y_test)


# ## Gradient Booster

# In[124]:



gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
gb_reg.fit(X_train_scaled, Y_train)
gb_pred = gb_reg.predict(X_test_scaled)

r2_score(Y_test, gb_pred)


# In[121]:


plt.figure(figsize=(10,5))
plt.title('Results for all models')
models = {"gb_reg":  GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1), "LogisticRegressionr": ridge_reg,"Forest regressor": RandomForestRegressor(n_estimators=100, max_depth=5)}

outcome = []

for model in models.values():
    kf = KFold(n_splits=6,random_state = 42,shuffle = True)
    cv_result = cross_val_score(model,X_train_scaled,Y_train,cv=kf)
    outcome.append(cv_result)
plt.boxplot(outcome,labels = models.keys())
plt.show()


# # Insights

# 1. The monthly supply of houses is the number of homes that are available for sale divided by the number of homes that are sold each month. 
# 2. A low monthly supply indicates that there are more buyers than sellers, which can lead to higher prices. 
# 3. Consumer sentiment: Consumer sentiment is a measure of how confident consumers are about the economy. 
# 4. A high level of consumer sentiment indicates that consumers are more likely to buy homes, which can lead to higher demand and higher prices. 
# 5. Housing inventory: Housing inventory is the number of homes that are available for sale. 
# 6. A low housing inventory indicates that there are fewer homes available for sale, which can lead to higher prices. 
# 7. Median sales price: The median sales price is the price at which half of the homes sold for more and half sold for less. 
# 8. A rising median sales price indicates that home prices are increasing. 
# 9. GDP: GDP is the total value of goods and services produced in a country. 
# 10. A rising GDP indicates that the economy is growing, which can lead to higher demand for housing and higher prices. 
# 11. As you can see, the supply and demand of houses are closely correlated. 
# 12. When there is a low supply of houses and a high demand, prices tend to rise. 
# 13. When there is a high supply of houses and a low demand, prices tend to fall.

# In[ ]:




