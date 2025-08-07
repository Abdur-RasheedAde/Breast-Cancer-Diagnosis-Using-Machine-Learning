#!/usr/bin/env python
# coding: utf-8

# ## Diagnose Breast Cancer
# 
# New notebook

# In[1]:


pip install seaborn


# In[2]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load data into pandas DataFrame from "/lakehouse/default/Files/3. Breast cancer dataset.csv"
df = pd.read_csv("/lakehouse/default/Files/3. Breast cancer dataset.csv")
display(df)


# In[4]:


df.head(10)


# In[5]:


# Drop unnecessary columns from the DataFrame
df = df.drop(columns=['id', 'Unnamed: 32'])

df.head(5)


# In[6]:


# Basic information about the dataset
print("\nBasic info of the dataset:")
df.info()


# In[7]:


#Check the unique values in column 'diagnosis'
df['diagnosis'].unique()


# In[8]:


#Map the unique values in 'diagnosis' column as 0 or 1
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})


# In[9]:


df.head(4)


# In[10]:


df.columns


# In[11]:


mean_features = list(df.columns[1:11])
se_features = list(df.columns[11:21])
worst_features = list(df.columns[21:31])


# In[12]:


print(se_features)


# In[13]:


#Append the 'diagnosis' to each of the list
mean_features.append('diagnosis')
se_features.append('diagnosis')
worst_features.append('diagnosis')


# In[14]:


#Check the correlations

corr = df[mean_features].corr
print(corr)


# In[15]:


# 1. Correlation Heatmap between 'diagnosis' and 'mean_features'
import matplotlib.pyplot as plt

corr = df[mean_features].corr

# Generate the heatmap using only the numerical columns
plt.figure(figsize=(10,6))
sns.heatmap(df[mean_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[16]:


# 2. Correlation Heatmap between 'diagnosis' and 'se_features'
import matplotlib.pyplot as plt

corr2 = df[se_features].corr

# Generate the heatmap using only the numerical columns
plt.figure(figsize=(10,6))
sns.heatmap(df[se_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[17]:


# 2. Correlation Heatmap between 'diagnosis' and 'worst_features'
import matplotlib.pyplot as plt

corr3 = df[worst_features].corr

# Generate the heatmap using only the numerical columns
plt.figure(figsize=(10,6))
sns.heatmap(df[worst_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[49]:


df.head(2)


# In[50]:


sns.pairplot(df, hue='radius_mean')
plt.savefig('radius_mean vs others.png')  # Save the figure as a PNG file
plt.show()


# ## **Training the Model**

# In[18]:


print(mean_features)
print(se_features)
print(worst_features)


# In[19]:


#Prediction variable
predictions_vars = ['radius_mean', 'perimeter_mean', 'area_mean','compactness_mean','concavity_mean', 'concave points_mean', 'radius_se', 'area_se', 'radius_worst', 'perimeter_worst', 'compactness_worst',]


# In[20]:


# Splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.15, random_state=1)


# In[21]:


#assigning variables for important features

train_x = train[predictions_vars]
train_y = train['diagnosis']
test_x = test[predictions_vars]
test_y = test['diagnosis']


# In[22]:


# Import models
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ##### **Random Forest**

# In[23]:


#fit model to traning dataset

model = RandomForestClassifier()

model.fit(train_x, train_y)


# In[28]:


#prediction
predictions = model.predict(test_x)


# In[25]:


test_y


# In[29]:


#evaluate performance of model

from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, predictions)


# In[30]:


#evaluate performance of model further more

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[32]:


precision = precision_score(test_y, predictions)
print("The precision is %.2f" % precision)

recall = recall_score (test_y, predictions)
print("The recall is %.2f" % recall)


# In[36]:


#evaluate performance of model further more

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_y, predictions)
print("The accuracy is %.2f" % accuracy)


# ##### **KNeighborsClassifier**

# In[34]:


#Try KNN and fit model to traning dataset

model = KNeighborsClassifier()
model.fit(train_x, train_y)


# In[35]:


#prediction
predictions = model.predict(test_x)

test_y


# In[38]:


#evaluate performance of model

from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, predictions)


# In[39]:


#evaluate performance of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

precision = precision_score(test_y, predictions)
recall = recall_score (test_y, predictions)
accuracy = accuracy_score(test_y, predictions)

print("The precision is %.2f" % precision)

print("The recall is %.2f" % recall)
print("The accuracy is %.2f" % accuracy)


# ##### **SVC**

# In[40]:


#Try KNN and fit model to traning dataset

model = SVC()
model.fit(train_x, train_y)

#prediction
predictions = model.predict(test_x)

test_y


# In[41]:


#evaluate performance of model

from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, predictions)


# In[42]:


#evaluate performance of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

precision = precision_score(test_y, predictions)
recall = recall_score (test_y, predictions)
accuracy = accuracy_score(test_y, predictions)

print("The precision is %.2f" % precision)

print("The recall is %.2f" % recall)
print("The accuracy is %.2f" % accuracy)


# ##### **MLPClassifier**

# In[43]:


#Try KNN and fit model to traning dataset

model = MLPClassifier()
model.fit(train_x, train_y)

#prediction
predictions = model.predict(test_x)

test_y


# In[44]:


#evaluate performance of model

from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, predictions)


# In[45]:


#evaluate performance of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

precision = precision_score(test_y, predictions)
recall = recall_score (test_y, predictions)
accuracy = accuracy_score(test_y, predictions)

print("The precision is %.2f" % precision)

print("The recall is %.2f" % recall)
print("The accuracy is %.2f" % accuracy)


# ##### **Search for the best model with Grid Search**
# 
# To pick the best model

# In[52]:


# Hyperparameter tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_x, train_y)

best_model = grid_search.best_estimator_

# Predictions with the best model
y_pred = best_model.predict(test_x)

# Calculate and print RMSE for the best model
rmse = np.sqrt(mean_squared_error(test_y, y_pred))
print(f"Optimized RMSE: {rmse:.2f}")

