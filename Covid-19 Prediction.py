#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENCE WITH PYTHON
# ### DATA SCIENCE PROJECT 
# ### CHRISTABEL GACHERI CHOMBA
# ##### SUPERVISED LEARNING: COVID-19 PREDICTION
# - creating an app that allows patient to input their symptoms and try to predict if they have COVID-19 or not
# - the link to the data: https://github.com/nshomron/covidpred/tree/master/data

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Importing Dataset
train  =  pd.read_csv('corona_tested_individuals_ver_006.english.csv')
train


# In[ ]:


train.info()


# ### Data cleaning

# In[4]:


train['cough'].unique()


# In[5]:


train['cough'].replace(
    {'0' : 0,
     '1' : 1,
     'None' : 0}, inplace = True
)
train['cough'].unique()


# In[6]:


train['fever'].replace({'1' : 1,
                        '0' : 0,
                        'None' : 0}, inplace = True)
train['fever'].unique()


# In[7]:


train['sore_throat'].replace({'0' : 0,
                              '1' : 1,
                              'None' : 0}, inplace = True)
train['sore_throat'].unique()


# In[8]:


train['head_ache'].replace({'1' : 1,
                            '0' : 0,
                            'None': 0}, inplace =True)
train['head_ache'].unique()


# In[9]:


train['shortness_of_breath'].replace({'0' : 0,
                                      '1' : 1,
                                      'None' : 0}, inplace = True)
train['shortness_of_breath'].unique()


# ### Data visualization
# - create visualizations to get insights from the data

# In[10]:


plt.figure(figsize = (10,5))
plt.subplot(121)
sns.countplot(x = train['fever'])
plt.xticks([0,1], ['No', 'Yes'])

plt.subplot(122)
plt.pie(x =list(train['fever'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('Fever pie chart')
plt.show()


# In[5]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.countplot(x = train['cough'])
plt.title('cough Countplot')
plt.xticks([1,0], ['Yes', 'No'])

plt.subplot(1,2,2)
plt.pie(x =list(train['cough'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('cough pie chart')
plt.show()


# In[6]:


plt.figure(figsize = (10,5))
plt.subplot(121)
sns.countplot(x = train['shortness_of_breath'])
plt.xticks([0,1], ['No', 'Yes'])

plt.subplot(122)
plt.pie(x =list(train['shortness_of_breath'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('Shortness of breathe pie chart')
plt.show()


# In[13]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.countplot(x = train['head_ache'])
plt.xticks([0,1], ['No', 'Yes'])

plt.subplot(1,2,2)
plt.pie(x =list(train['head_ache'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('Head Ache pie chart')
plt.show()


# In[14]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.countplot(x = train['sore_throat'])
plt.xticks([0,1], ['No', 'yes'])

plt.subplot(1,2,2)
plt.pie(x =list(train['sore_throat'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('sore throat pie chart')
plt.show()


# In[15]:


list(train['corona_result'].value_counts())


# In[16]:


plt.figure(figsize = (10,5))
plt.subplot(121)
sns.countplot(train['corona_result'])

plt.subplot(1,2,2)
plt.pie(x =list(train['corona_result'].value_counts()), labels = ['Negative', 'Positive', 'other'], autopct = '%0.1f%%')
plt.title('Corona Results pie chart')
plt.show()


# In[17]:


sns.countplot(x = train['gender'])
plt.show()


# #### Mask the data and get only the positive results

# In[18]:


positive = train.where(train['corona_result'] == 'positive').dropna()
positive.head()


# In[19]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.countplot(x = positive['sore_throat'])
plt.xticks([0,1], ['No', 'yes'])

plt.subplot(1,2,2)
plt.pie(x =list(positive['sore_throat'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('sore throat pie chart')
plt.show()


# In[20]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.countplot(x = positive['head_ache'])
plt.xticks([0,1], ['No', 'yes'])

plt.subplot(1,2,2)
plt.pie(x =list(positive['head_ache'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('Headache pie chart')
plt.show()


# In[21]:


plt.figure(figsize = (10,5))
plt.subplot(121)
sns.countplot(x = positive['shortness_of_breath'])
plt.xticks([0,1], ['No', 'Yes'])

plt.subplot(122)
plt.pie(x =list(positive['shortness_of_breath'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('Shortness of breathe pie chart')
plt.show()


# In[22]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.countplot(x = positive['cough'])
plt.title('cough Countplot')
plt.xticks([1,0], ['Yes', 'No'])

plt.subplot(1,2,2)
plt.pie(x =list(positive['cough'].value_counts()), labels = ['No', 'Yes'], autopct = '%0.1f%%')
plt.title('cough pie chart')
plt.show()


# #### Visualization insights
# - Cough was the most common symptom reported by **15.1%** of the participants
# - Fever was reported by **7.8%** of the study participants
# - shortness of breath was the least common symptom being reported by **0.6%** of the participants
# - Sore throat was reported by only **0.7%** of the study participants
# - **5.4%** of the tested returned positive corona virus results while **94.6%** returned negative results
# - **10.4%** of the positive results reported sore throat as a symptom
# - **15.2%** of the positive results reported having headaches
# - **7.9%** of the positive results reported experiencing shortness of breathe
# - coughing was the most common symptom reported by **44.7%** of all the positive cases

# In[23]:


train.info()


# In[24]:


#convert the test_date column to datetime
train['test_date'] = pd.to_datetime(train['test_date'])
train.info()


# In[25]:


#check the unique entries in the corona results column
train['corona_result'].unique()


# In[26]:


#change the 'other' results to null and dropping the data

train['corona_result'].replace({'other' : np.nan}, inplace = True)
len(train)


# In[27]:


train.info()


# In[28]:


no_null = train.dropna()
no_null.info()


# In[29]:


# change the corona results to binary data for training
no_null['corona_result'].replace({'positive' : 1,
                                  'negative' : 0}, inplace = True)
no_null


# group by date and get a time series plot
# 

# In[30]:


grouped = no_null.groupby('test_date').agg({'corona_result' : np.sum})
grouped.reset_index(inplace = True)
grouped.head()


# In[31]:


plt.figure(figsize = (15, 8))
plt.plot(grouped['test_date'], grouped['corona_result'], '--', color = 'orange')
plt.xlabel('Date')
plt.ylabel('No of Cases')
plt.title('Covid-19 Cases')
plt.xticks(rotation = 0)
plt.show()


# ### Data modelling
# - performing supervised machine learning on the dataset
# 

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[33]:


x = no_null[['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache']]
y = no_null['corona_result']


# #### Perform a train_test split on our data

# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state = 0)


# In[35]:


from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression().fit(X_train, Y_train) 
predicts_log = clf_log.predict(X_test)
accuracy = accuracy_score(predicts_log, Y_test)
print(accuracy)


# In[36]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

print(f'precision score : {precision_score(predicts_log, Y_test)}')
print(f'Recall score : {recall_score(predicts_log, Y_test)}')
print('\n\n Confusion Matrix : ')
confusion_matrix(predicts_log, Y_test)


# ### Train a suppor vector machine and evaluate its performance

# In[37]:


from sklearn.svm import SVC
clf_svc = SVC(kernel = 'linear', gamma = 'scale').fit(X_train, Y_train)
predicts_svc = clf_svc.predict(X_test)
predicts_train = clf_svc.predict(X_train)

print(f'accuracy score on training data : {accuracy_score(predicts_train, Y_train)}')
print(f'precision score on training data : {precision_score(predicts_train, Y_train)}')
print(f'recall score on training data : {recall_score(predicts_train, Y_train)} \n\n')

print(f'accuracy score on test data : {accuracy_score(predicts_svc, Y_test)}')
print(f'precision score on test data : {precision_score(predicts_svc, Y_test)}')
print(f'recall score on test data : {recall_score(predicts_svc, Y_test)}')
print(f'\n\n Confusion Matrix : \n {confusion_matrix(predicts_svc, Y_test)}')


# ### Train a Decision Tree Classifier

# In[38]:


from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier().fit(X_train, Y_train)
predicts_dt = clf_dt.predict(X_test)
predicts_train = clf_dt.predict(X_train)

print(f'accuracy score on training data : {accuracy_score(predicts_train, Y_train)}')
print(f'precision score on training data : {precision_score(predicts_train, Y_train)}')
print(f'recall score on training data : {recall_score(predicts_train, Y_train)} \n\n')

print(f'accuracy score on test data : {accuracy_score(predicts_dt, Y_test)}')
print(f'precision score on test data : {precision_score(predicts_dt, Y_test)}')
print(f'recall score on test data : {recall_score(predicts_dt, Y_test)}')
print(f'\n\n Confusion Matrix : \n {confusion_matrix(predicts_dt, Y_test)}')


# ### Train a Random Forest Classifier

# In[40]:


from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier().fit(X_train, Y_train)
predicts_rfc = clf_rfc.predict(X_test)
predicts_train = clf_rfc.predict(X_train)

print(f'accuracy score on training data : {accuracy_score(predicts_train, Y_train)}')
print(f'precision score on training data : {precision_score(predicts_train, Y_train)}')
print(f'recall score on training data : {recall_score(predicts_train, Y_train)} \n\n')

print(f'accuracy score on test data : {accuracy_score(predicts_rfc, Y_test)}')
print(f'precision score on test data : {precision_score(predicts_rfc, Y_test)}')
print(f'recall score on test data : {recall_score(predicts_rfc, Y_test)}')
print(f'\n\n Confusion Matrix : \n {confusion_matrix(predicts_rfc, Y_test)}')


# ### Import a Gradient Boosted Classifier and fit the training data

# In[42]:


from sklearn.ensemble import GradientBoostingClassifier
clf_gbc = GradientBoostingClassifier().fit(X_train, Y_train)
predicts_gbc = clf_gbc.predict(X_test)
predicts_train = clf_gbc.predict(X_train)

print(f'accuracy score on training data : {accuracy_score(predicts_train, Y_train)}')
print(f'precision score on training data : {precision_score(predicts_train, Y_train)}')
print(f'recall score on training data : {recall_score(predicts_train, Y_train)} \n\n')

print(f'accuracy score on test data : {accuracy_score(predicts_gbc, Y_test)}')
print(f'precision score on test data : {precision_score(predicts_gbc, Y_test)}')
print(f'recall score on test data : {recall_score(predicts_gbc, Y_test)}')
print(f'\n\n Confusion Matrix : \n {confusion_matrix(predicts_gbc, Y_test)}')


# ### Evaluate all the models aganist a Dummy Classifier

# In[44]:


from sklearn.dummy import DummyClassifier
clfdummy = DummyClassifier('most_frequent').fit(X_train, Y_train)
predicts_dummy = clfdummy.predict(X_test)
predicts_train = clfdummy.predict(X_train)

print(f'accuracy score on training data : {accuracy_score(predicts_train, Y_train)}')
print(f'precision score on training data : {precision_score(predicts_train, Y_train)}')
print(f'recall score on training data : {recall_score(predicts_train, Y_train)} \n\n')

print(f'accuracy score on test data : {accuracy_score(predicts_dummy, Y_test)}')
print(f'precision score on test data : {precision_score(predicts_dummy, Y_test)}')
print(f'recall score on test data : {recall_score(predicts_dummy, Y_test)}')


# In[ ]:




