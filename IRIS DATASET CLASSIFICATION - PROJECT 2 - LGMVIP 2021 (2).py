#!/usr/bin/env python
# coding: utf-8

# # About the project 
# 
# IRIS DATASET - A multivariant datset used for machine learning purposes.The following dataset contains a set of 150 records under five attributes 
# - sepal length 
# - sepal width
# - petal length
# - petal width
# - species
# In this data set we analyize three species of Iris flower, i-e Iris setosa , Iris versicolor , and Iris verginica.

# # 1st step = load the image
# 
# In this step you have to load the images which give you the visuals of classes contain in the Iris dataset.For this step , download the image from google and save in the directory in which your jupyter file is present. 

# In[30]:


from PIL import Image
Irisflower = Image.open('iris.png')
Irisflower


# # 2nd step = Basic Libraries ( Loading ) 
# 
# we import libraries to perform specific tasks as every library is specified to do certain task.First off , before importing the libraries you have to install libraries. You can download any library via command prompt ( cmb ).Open the cmb , open the directory where your python is present and installed , then open the script file and using pip command download individual libraries. 
# Use the below code mechanism to import the libraries.

# In[31]:


import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# # 3rd step = Dataset ( Loading )
# 
# In this step , before loading the dataset it is important to download the dataset.You can easily download the dataset from internet search. Then save the dataset csv file in the same directory from where you launched the jupyter notebook.Its one way to load the dataset in the jupyter notebook. You can also load the data from sklearn library but the mannual way of donloading the dataset and then loading is quite easy and comprehensive for the beginners. 
# 
# Use the following code mechanism to load the dataset.

# In[32]:


data = pd.read_csv("iris.csv")


# # 4th step = Dataset Information expansion

# In[33]:


#to show the first five rows of the dataset.Use .head() command.sometimes the dataset contain the column ID. You can remove the ID column by simple command .drop(column).
data.head()


# In[11]:


#In order to show the stats about the dataset
data.describe()


# In[34]:


#In order to display the basic info about the data type 
data.info()


# In[35]:


#to show the no of samples on each class
data['species'].value_counts()


# # 5th step = Dataset preprocessing

# In[36]:


#check for null values. If in case null values are present we can replace it.
data.isnull().sum()


# # 6th step =  Dataset ( Exploratory Analysis ) 
# 
# This analysis is to convert the given dataset into graphical form.

# In[37]:


#to create histograms for each set of dataset
data['sepal_length'].hist()


# In[27]:


data['sepal_width'].hist()


# In[38]:


data['petal_length'].hist()


# In[39]:


data['petal_width'].hist()


# In[40]:


#to create scatter plot. Use the below code mechanism
colors = ['yellow' , 'purple' , 'red']
species = ['setosa' , 
'versicolor' , 
'virginica' ]


# In[41]:


for i in range(3):
    x = data[data['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['sepal_width'], c=colors[i], label=species[i])
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend()  


# In[42]:


for i in range(3):
    x = data[data['species'] == species[i]]
    plt.scatter(x['petal_length'], x['petal_width'], c=colors[i], label=species[i])
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend() 


# In[43]:


for i in range(3):
    x = data[data['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['petal_length'], c=colors[i], label=species[i])
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend() 


# # 7th Matrix correlation of dataset
# 
# A correlation matrix is a table which shows us the variables along with co-efficients. the individual cell in the table shows the correlation between two variables. values range ( -1 to 1 ). if two variables have high correlation , we can neglect one variable from those two.

# In[44]:


data.corr()


# In[53]:


corr = data.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm') 

#we can change the size of the figure by changing the numbers in the 2nd code line.


# # 8th step = Label encoding of dataset
# 
# In machine learning , we usually deal with datasets which contain multiple labels in one or more than one columns. the process of label encoding is use to convert the labels into numeric form.

# In[46]:


#install sklearn and load it in the same directory.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[47]:


data['species'] = le.fit_transform=(data['species'])
data.head()


# # 9th step = Model training of dataset 

# In[48]:


from sklearn.model_selection import train_test_split
#train - 70
#test - 30
X = data.drop(columns=['species'])
Y = data['species']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.2)


# In[49]:


#logistisc regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[55]:


#model trainning
model.fit(x_train, y_train)


# # 10th step = Performance checking of dataset

# In[51]:


#print matric to get performance
print("Accuracy:", model.score(x_test, y_test)*100)

#based on the test and train dataset the accuracy varies.


# In[54]:





# In[ ]:





# In[ ]:




