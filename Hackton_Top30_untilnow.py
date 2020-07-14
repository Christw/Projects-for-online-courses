#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re
from collections import Counter
from statistics import mode
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from scipy import stats
import datetime as dt


# Data wrangling 

# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train.head()


# In[4]:


train.shape


# In[5]:


test.head()


# In[6]:


test.shape


# In[7]:


train.describe(include=['object'])


# In[8]:


#date>num for train
train['Num_Date'] = pd.to_datetime(train['Date'])
train['Num_Date']=train['Num_Date'].map(dt.datetime.toordinal)


# In[9]:


import datetime as dt


#month/year/day transform
train['Date'] = pd.to_datetime(train['Date'])
train.set_index('Date', inplace=True)
train.sort_index(inplace=True)
train['month'] = train.index.month
train['year'] = train.index.year
train['day'] = train.index.day
train.head()


# In[10]:


train.describe()


# In[11]:


train["Demand"].sort_values(ascending=False)
k=train[train.Demand<10]
k["Demand"].value_counts()


# In[12]:


bins = [0, 3, 500, 25000]


# In[13]:


#Demand can be categoried cuz it seems to related our topics
#Need to delete outlier!
#How to define low,medium,high demands? to fit what the context mentioned, like particular goods are needed when the price is low
group_names = ['Low','Medium','High']
train['Demand-binned'] = pd.cut(train['Demand'], bins, labels=group_names, include_lowest=True )


train['Demand-binned'].value_counts()


# In[14]:


#Grade can be categroied too.
train['Grade_Obj']=train["Grade"]
k=pd.get_dummies(train['Grade_Obj'],prefix="Grade")
train=train.join(k)
train.head()


# In[15]:


#Product cate can try?
train["Product_Category"].value_counts()


# In[16]:


#Market cate can try too?
train["Market_Category"].value_counts()
train["Market_Category_Obj"]=train["Market_Category"].astype(object)
train["State_of_Country_Obj"]=train["State_of_Country"].astype(object)
train["Product_Category_Obj"]=train["Product_Category"].astype(object)
train['Grade_Obj']=train['Grade_Obj'].astype(object)
train.head()


# In[17]:


# grouping results for demand vs. market_cate
train_test = train[['Demand-binned','Market_Category_Obj','Low_Cap_Price']]
grouped_test1 = train_test.groupby(['Demand-binned','Market_Category_Obj'],as_index=False).mean()
grouped_test1


# In[18]:


grouped_pivot = grouped_test1.pivot(index='Demand-binned',columns='Market_Category_Obj')
grouped_pivot


# In[19]:


train.head(5)


# EDA

# In[20]:


train.dtypes


# In[21]:


fig, axs = plt.subplots(nrows=1, figsize=(13, 13))
sns.heatmap(train.corr(), annot=True, square=True, cmap='YlGnBu', linewidths=2, linecolor='black', annot_kws={'size':12})


# Based on the figures above, there are a few tests we need to perform to obtain the important variables.
# Categorical: Grade,Demand,Market,State,Product>>ANOVA 
# Numerious:High_cap_price>>Pearson test
# 6 tests in total!

# In[22]:


#High_price vs Low_price
pearson_coef, p_value = stats.pearsonr(train['Low_Cap_Price'], train['High_Cap_Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
# high okay!


# In[23]:


#Demand vs Low_price
grouped_test2=train[['Demand-binned', 'Low_Cap_Price']].groupby(['Demand-binned'])
grouped_test2.get_group('Low')['Low_Cap_Price']

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('Low')['Low_Cap_Price'], grouped_test2.get_group('Medium')['Low_Cap_Price'],grouped_test2.get_group('High')['Low_Cap_Price'])
 
print( "ANOVA results: F=", f_val, ", P =", p_val) 

f_val1, p_val1 = stats.f_oneway(grouped_test2.get_group('Low')['Low_Cap_Price'], grouped_test2.get_group('Medium')['Low_Cap_Price'])
 
print( "ANOVA results: F=", f_val1, ", P =", p_val1) 
f_val2, p_val2 = stats.f_oneway(grouped_test2.get_group('Low')['Low_Cap_Price'], grouped_test2.get_group('High')['Low_Cap_Price'],grouped_test2.get_group('High')['Low_Cap_Price'])
 
print( "ANOVA results: F=", f_val2, ", P =", p_val2) 
f_val3, p_val3 = stats.f_oneway(grouped_test2.get_group('Low')['Low_Cap_Price'], grouped_test2.get_group('High')['Low_Cap_Price'],grouped_test2.get_group('High')['Low_Cap_Price'])
 
print( "ANOVA results: F=", f_val3, ", P =", p_val3) 
#Demand okay


# In[24]:


train["Grade_Obj"].value_counts()


# In[25]:


#Grade vs Low_price

bins = [-1, 0.5, 1.5, 2.2, 3.3]
group_names = ['Low','Medium_lower','Medium_higher','High']
train['Grade_Obj'] = pd.cut(train['Grade'], bins, labels=group_names, include_lowest=True )
grouped_test2=train[['Grade_Obj', 'Low_Cap_Price']].groupby(['Grade_Obj'])

# ANOVA
f_val4, p_val4 = stats.f_oneway(grouped_test2.get_group('Low')['Low_Cap_Price'], grouped_test2.get_group('Medium_lower')['Low_Cap_Price'],grouped_test2.get_group('Medium_higher')['Low_Cap_Price'],grouped_test2.get_group('High')['Low_Cap_Price'])
 
print( "ANOVA results: F=", f_val4, ", P =", p_val4) 

#Grade okay!


# In[26]:


#Feature Engineering
# 1. For State_of_Country
k=pd.get_dummies(train['State_of_Country'],prefix="State_of_Country")
result = pd.concat([train, k], axis=1)
result
# 2. For Product_cat
k1=pd.get_dummies(train['Product_Category'],prefix="Product_Category")
result = pd.concat([result, k1], axis=1)


# In[27]:


train=train.drop(['Item_Id','Demand-binned', 'Grade_Obj', 'Grade_0', 'Grade_1',
       'Grade_2', 'Grade_3', 'Market_Category_Obj', 'State_of_Country_Obj',
       'Product_Category_Obj'], axis=1)


# In[28]:


p=train.drop(['Low_Cap_Price'], axis=1)


# In[29]:


p.reset_index(inplace=True) # Resets the index, makes factor a column
p.drop("Date",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True


# In[30]:


p.shape
p=p.drop_duplicates()


# In[31]:


#date>num for train
test['Num_Date'] = pd.to_datetime(test['Date'])
test['Num_Date']=test['Num_Date'].map(dt.datetime.toordinal)


# In[32]:


#month/year/day transform for test
test['Date'] = pd.to_datetime(test['Date'])
test.set_index('Date', inplace=True)
test.sort_index(inplace=True)
test['month'] = test.index.month
test['year'] = test.index.year
test['day'] = test.index.day
test.head()
j=test


# In[33]:


j.reset_index(inplace=True) # Resets the index, makes factor a column
j.drop("Date",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True


# In[34]:


o=train
o.reset_index(inplace=True) # Resets the index, makes factor a column
o.drop("Date",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True
o=o.drop_duplicates()
o.head()


# In[35]:


from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor()
model.fit(p,o.Low_Cap_Price)
new_test=j.drop(['Item_Id'], axis=1)
y_test=model.predict(new_test)
print(model.score(p,o.Low_Cap_Price))
print(cross_val_score(model,p,o.Low_Cap_Price,cv=5).mean())


# In[36]:


from sklearn.ensemble import GradientBoostingRegressor


model2=GradientBoostingRegressor()
model2.fit(p,o.Low_Cap_Price)
y_test2=model2.predict(new_test)
print(model2.score(p,o.Low_Cap_Price))
print(cross_val_score(model2,p,o.Low_Cap_Price,cv=5).mean())


# In[37]:


from sklearn.linear_model import LinearRegression
model3=LinearRegression()
model3.fit(p,o.Low_Cap_Price)
y_test3=model3.predict(new_test)
print(model3.score(p,o.Low_Cap_Price))
print(cross_val_score(model3,p,o.Low_Cap_Price,cv=5).mean())


# In[38]:


p_1=p.drop(['State_of_Country','Market_Category','Product_Category'], axis=1)
model4=LinearRegression()
model4.fit(p_1,o.Low_Cap_Price)
g=new_test.drop(['State_of_Country','Market_Category','Product_Category'], axis=1)
y_test4=model4.predict(g)
print(model4.score(p_1,o.Low_Cap_Price))
print(cross_val_score(model4,p_1,o.Low_Cap_Price,cv=5).mean())


# In[39]:


from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(p,o.Low_Cap_Price)
# make predictions
predictions = my_model.predict(new_test)
print(my_model.score(p,o.Low_Cap_Price))
print(cross_val_score(my_model,p,o.Low_Cap_Price,cv=5).mean())


# In[41]:


test["Low_Cap_Price"]=pd.DataFrame(data=y_test2)


# In[42]:


test.head()


# In[47]:


submission = pd.DataFrame({'Item_Id':test['Item_Id'],'Low_Cap_Price':test['Low_Cap_Price']})


submission.to_csv('predict_the_lowest_price',index=False)


# In[ ]:




