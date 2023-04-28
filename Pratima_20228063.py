#!/usr/bin/env python
# coding: utf-8

# In[42]:


# Importing the libraries
import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[43]:


#Loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[44]:


print(breast_cancer_dataset)


# In[45]:


#loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)


# In[46]:


data_frame.head()


# In[47]:


#Adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target


# In[48]:


data_frame.shape


# In[49]:


data_frame.info()


# In[50]:


data_frame.tail()


# In[51]:


data_frame.describe()


# In[52]:


data_frame.boxplot()
plt.show()


# In[53]:


data_frame.hist(bins=100, figsize=(10,15,))
plt.show()


# In[54]:


plt.figure(figsize=(10,7))
sns.heatmap(data_frame.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[29]:


# Checking the distribution of Terget Variable
data_frame['label'].value_counts()

# 1 means Benign
# 0 means Malignant


# In[30]:


#Separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']


# In[31]:


print(X)


# In[32]:


# Spliting the data into training data and testing data. 80% training data and 20% testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, random_state=2)


# In[33]:


print(X.shape, X_train.shape, X_test.shape)


# # Logistic Regression

# In[34]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
logmodel = LogisticRegression()
logmodel.fit(X, Y)
print('#### Output of Logistic Regression using sklearn ###')
print('Coefficients:', logmodel.coef_)
print('Intercept:', logmodel.intercept_)


# In[35]:


logmodel = LogisticRegression()
logmodel.fit(X_train, Y_train)
Y_pred=logmodel.predict(X_test)
probs = logmodel.predict_proba(X_test)
prebs = probs[:,1]


# In[36]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('# Logistic Reg #')
print("confusion matrix")
print(confusion_matrix(Y_test, Y_pred))
accuracy_Logit = accuracy_score(Y_test,Y_pred)
print("Accuracy Score:",accuracy_Logit)


# In[37]:


input_data = (19.69,21.25,130.00,1203.0,0.10960,0.15990,0.1974,0.12790,0.2069,0.05999,0.7456,0.7869,4.585,94.03,0.006150,0.04006,0.03832,0.02058,0.02250,0.004571,23.57,25.53,152.50,1709.0,0.1444,0.4245,0.4504,0.2430,0.3613,0.08758)

# Change the input data to a numpy array
input_data_as_numpy_array = np.array(input_data)

#reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = logmodel.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print('The breast cancer is Malignant')
    
else:
    print('The breast cancer is Benign')


# # Random Forest

# In[38]:


from sklearn.ensemble import RandomForestClassifier
RFclf=RandomForestClassifier(n_estimators=101)
RFclf.fit(X_train, Y_train)
Y_predRF=RFclf.predict(X_test)
#Predict probabilities for the test data.
probsRF = RFclf.predict_proba(X_test)
#Keep Probabilities of the positive class only.
probsRF = probsRF[:, 1]


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,recall_score
print('# Random Forest #')
print("confusion matrix")
print(confusion_matrix(Y_test, Y_predRF))
accuracy_RF = accuracy_score(Y_test,Y_predRF)
print("Accuracy Score:",accuracy_RF)


# In[ ]:


By observing the accuracy score , Random forest model is better model for predicting the dataset.

