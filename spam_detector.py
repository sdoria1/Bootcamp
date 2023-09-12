#!/usr/bin/env python
# coding: utf-8

# # Spam Detector

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ## Retrieve the Data
# 
# The data is located at [https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv](https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv)
# 
# Dataset Source: [UCI Machine Learning Library](https://archive-beta.ics.uci.edu/dataset/94/spambase)
# 
# Import the data using Pandas. Display the resulting DataFrame to confirm the import was successful.

# In[2]:


# Import the data
data = pd.read_csv("https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv")
data.head()


# ## Predict Model Performance
# 
# You will be creating and comparing two models on this data: a Logistic Regression, and a Random Forests Classifier. Before you create, fit, and score the models, make a prediction as to which model you think will perform better. You do not need to be correct! 
# 
# Write down your prediction in the designated cells in your Jupyter Notebook, and provide justification for your educated guess.

# Detecting spam is a complex problem which considers a number of features, therefore a simple logistic regression, while powerful many circumstances, will likely be unable to handle this data set as well as a random forest.
# 
# Therefore, I predict that the random forest classifier will be better suited in this case.

# ## Split the Data into Training and Testing Sets

# In[3]:


# Create the labels set `y` and features DataFrame `X`

y = data['spam']
X = data.copy()
X = X.drop(columns = ['spam'])


# In[4]:


# Check the balance of the labels variable (`y`) by using the `value_counts` function.

y.value_counts()


# In[5]:


# Split the data into X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)


# ## Scale the Features

# Use the `StandardScaler` to scale the features data. Remember that only `X_train` and `X_test` DataFrames should be scaled.

# In[6]:


from sklearn.preprocessing import StandardScaler

# Create the StandardScaler instance

scaler = StandardScaler()


# In[7]:


# Fit the Standard Scaler with the training data

X_scaler = scaler.fit(X_train)


# In[8]:


# Scale the training data

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ## Create and Fit a Logistic Regression Model
# 
# Create a Logistic Regression model, fit it to the training data, make predictions with the testing data, and print the model's accuracy score. You may choose any starting settings you like. 

# In[10]:


# Train a Logistic Regression model and print the model score
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(random_state=1)

logistic_model.fit(X_train_scaled, y_train)

print(f"Training Data Score: {logistic_model.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {logistic_model.score(X_test_scaled, y_test)}")


# In[11]:


# Make and save testing predictions with the saved logistic regression model using the test data

predictions = logistic_model.predict(X_test_scaled)

# Review the predictions
predictions


# In[12]:


# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
acc_score = accuracy_score(y_test, predictions)

print(f"Accuracy Score is : {acc_score}")


# ## Create and Fit a Random Forest Classifier Model
# 
# Create a Random Forest Classifier model, fit it to the training data, make predictions with the testing data, and print the model's accuracy score. You may choose any starting settings you like. 

# In[13]:


# Train a Random Forest Classifier model and print the model score
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 128, random_state = 1)

rf_model.fit(X_train_scaled, y_train)

print(f"Training Data Score: {rf_model.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {rf_model.score(X_test_scaled, y_test)}")


# In[14]:


# Make and save testing predictions with the saved logistic regression model using the test data
predictions_rf = rf_model.predict(X_test_scaled)

# Review the predictions
predictions_rf


# In[15]:


# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
acc_score_rf = accuracy_score(y_test, predictions_rf)

# Display results
print(f"Accuracy Score is: {acc_score_rf}")


# ## Evaluate the Models
# 
# Which model performed better? How does that compare to your prediction? Write down your results and thoughts in the following markdown cell.

# As I predicted, the random forest model performed better, though interestingly it showed more signs of overfitting (note the difference between the training and testing scores) than the logistic model, which is not as I predicted.
# 
# This is likely because the logistic regression model is simpler than the random forest model, allowing the random forest model to train on variance in the testing data more readily.

# In[ ]:




