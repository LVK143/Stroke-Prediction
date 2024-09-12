#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv(r"C:\Users\Lankala Vinay Kumar\framingham.csv")
df.head()


# MALE: GENDER
# AGE: age of the patient continous
# Behaviour:
# currect smoker: Yes or not whether he is smoker or not
# cigsPerday: no of cig per day continous
# BPmeds:whether the patient is on BP meds or not nominal
# prevalen storke: whether the patient has store or not  nominal
# prevalenthyp:hypertensive  nominal 
# diabetes:he is diabet or not  nominal
# totChol: chol level (continous)
# sysBP: Systolic bp  (continous) 
# diaBP:dialostic bp (continous)
# BMI :body mass index (continous) 
# heartrate: no of beats per sec (continous)
# glucose: level of glucose in body 
#   
# Predictive var
# 
# Ten Year CHD:
#     
#     whether the person  will be heart attack or not
#     

# In[5]:


df.info()


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[13]:


df.isnull().sum()


# In[14]:


df.isnull().mean()*100


# In[ ]:


df.drop(columns=['education'],axis=1,inplace=True)


# In[19]:


df


# In[20]:


df.isnull().sum()


# In[23]:


df['cigsPerDay'] = df['cigsPerDay'].fillna(df['cigsPerDay'].mean())
df['cigsPerDay'].isnull().sum()


# In[24]:


df['BPMeds'] = df['BPMeds'].fillna(df['BPMeds'].median())
df['BPMeds'].isnull().sum()


# In[26]:


df['totChol'] = df['totChol'].fillna(df['totChol'].median())
df['totChol'].isnull().sum()


# In[25]:


df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
df['BMI'].isnull().sum()


# In[28]:


df['heartRate'] = df['heartRate'].fillna(df['heartRate'].median())
df['heartRate'].isnull().sum()


# In[29]:


df['glucose'] = df['glucose'].fillna(df['glucose'].median())
df['glucose'].isnull().sum()


# In[30]:


df.isnull().sum()


# In[31]:


df.describe()


# In[32]:


df.corr()


# # HEAT MAP

# A heatmap is a great way to visualize the correlation matrix, as it provides a clear and intuitive graphical representation of the relationships between variables. Here's how you can create a heatmap using Python with the seaborn library:
# 
# 

# Explanation
# DataFrame Initialization: The correlation matrix is initialized as a DataFrame, where each variable is a column and an index.
# 
# Heatmap Plotting: The sns.heatmap function is used to create the heatmap:
# 
# annot=True: Annotates each cell with the numeric value.
# fmt='.2f': Formats the annotation text to two decimal places.
# cmap='coolwarm': Uses the 'coolwarm' colormap, which ranges from cool colors (blue) for negative values to warm colors (red) for positive values.
# cbar=True: Displays the color bar.
# square=True: Makes each cell square-shaped.
# Displaying the Plot: plt.show() is called to display the heatmap.
# 
# This code will generate a heatmap that visually represents the correlations between the variables in your dataset, making it easier to identify strong, moderate, and weak relationships.
# 
# 
# 
# 
# 
# 
# 

# In[40]:


\


# The table  which shows the pairwise correlation coefficients between different variables in a dataset. Correlation coefficients measure the strength and direction of the linear relationship between two variables. They range from -1 to 1, where:
# 
# 1 indicates a perfect positive linear relationship.
# -1 indicates a perfect negative linear relationship.
# 0 indicates no linear relationship.
# How to Interpret the Correlation Matrix
# Diagonal Values: All diagonal values are 1 because a variable is always perfectly correlated with itself.
# 
# Positive Correlation: Values closer to 1 indicate a strong positive relationship, meaning as one variable increases, the other tends to increase as well. For example:
# 
# male and cigsPerDay have a correlation of 0.316807, suggesting that males tend to smoke more cigarettes per day.
# age and sysBP have a correlation of 0.394302, indicating that older age is associated with higher systolic blood pressure.
# Negative Correlation: Values closer to -1 indicate a strong negative relationship, meaning as one variable increases, the other tends to decrease. For example:
# 
# currentSmoker and age have a correlation of -0.213748, suggesting that older people are less likely to be current smokers.
# male and heartRate have a correlation of -0.116621, indicating that males tend to have a slightly lower heart rate.
# Weak or No Correlation: Values close to 0 indicate a weak or no linear relationship. For example:
# 
# diabetes and prevalentStroke have a correlation of 0.006949, indicating almost no linear relationship between having diabetes and having a prevalent stroke.
# male and prevalentStroke have a correlation of -0.004546, indicating almost no relationship between being male and having a prevalent stroke.
# Identifying Relationships
# To identify important relationships in your data:
# 
# Strong Relationships: Look for correlation values close to 1 or -1.
# 
# For example, sysBP and diaBP have a strong positive correlation of 0.784002, indicating that systolic and diastolic blood pressures are strongly related.
# Moderate Relationships: Correlation values between 0.3 and 0.7 (or -0.3 and -0.7) suggest moderate relationships.
# 
# For instance, prevalentHyp and sysBP have a correlation of 0.696755, showing a moderately strong positive relationship.
# Weak Relationships: Correlation values between 0.1 and 0.3 (or -0.1 and -0.3) suggest weak relationships.
# 
# For example, age and TenYearCHD have a correlation of 0.225256, indicating a weak positive relationship between age and the ten-year risk of coronary heart disease.
# Non-Relationships: Values close to 0 suggest no linear relationship.
# 
# For example, glucose and prevalentStroke have a correlation of 0.018722, indicating no significant linear relationship between glucose levels and having a prevalent stroke.
# Summary of Key Relationships in Your Data
# Strong Positive Correlations:
# 
# sysBP and diaBP (0.784002)
# prevalentHyp and sysBP (0.696755)
# currentSmoker and cigsPerDay (0.766970)
# Moderate Positive Correlations:
# 
# age and sysBP (0.394302)
# prevalentHyp and diaBP (0.615751)
# Negative Correlations:
# 
# currentSmoker and age (-0.213748)
# male and heartRate (-0.116621)
# When analyzing the data, it's crucial to consider the context and the possibility of confounding factors. Correlation does not imply causation, and further analysis is often needed to understand the underlying relationships.
# 
# 
# 
# 
# 
# 
# 

# In[45]:


X= df.iloc[:,:-1]
y = df['TenYearCHD']


# In[46]:


X.columns # TenYearCHD columns is excluded from the data set as it is output var 


# TRAIN_TEST_SPLIT

# In[48]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.2,random_state=2)


# In[55]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[57]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[59]:


y_pred = model.predict(X_test)
y_pred


# In[61]:


print("Model score of training data:",model.score(X_train,y_train)*100)
print("Model score of testing data:",model.score(X_test,y_test)*100)


# In[64]:


from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score,accuracy_score


# In[65]:


model.predict_proba(X_test)[:,1]


# In[66]:


model.predict_proba(X_test)[:,0]


# In[69]:


cf=confusion_matrix(y_test,y_pred)
cf


# In[70]:


sns.heatmap(cf,annot=True)
plt.show()


# In[71]:


accuracy_score(y_test,y_pred)


# In[73]:


precision_score(y_test,y_pred)*100


# In[74]:


recall_score(y_test,y_pred)*100


# data shows very poor  values in confusion matrix
# True Positives (TP): 3
# True Negatives (TN): 703
# False Positives (FP): 4
# False Negatives (FN): 138

# In[77]:


data = df.copy()
data.head()


# In[78]:


data.info()


# In[79]:


data.isnull().sum()


# In[80]:


data['TenYearCHD'].value_counts()


# In[81]:


data.columns


# In[121]:


X=data.iloc[:,:-1]
y = data['TenYearCHD']


# In[125]:


from imblearn.under_sampling import RandomUnderSampler
ru = RandomUnderSampler()
ru_X,ru_y = ru.fit_resample(X,y)


# In[127]:


ru_y.value_counts()


# In[128]:


X_train ,X_test,y_train,y_test = train_test_split(ru_X,ru_y,test_size=0.2,random_state=21)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[130]:


LR = LogisticRegression()
LR.fit(X_train,y_train)


# In[131]:


y_pred = LR.predict(X_test)
y_pred


# In[132]:


confusion_matrix(y_test,y_pred)


# In[149]:


# Confusion Matrix values
TN_under, FP_under, FN_under, TP_under = 86, 42, 54, 76

# Calculations
accuracy_under = (TP_under + TN_under) / (TP_under + TN_under + FP_under + FN_under)
precision_under = TP_under / (TP_under + FP_under)
recall_under = TP_under / (TP_under + FN_under)
f1_score_under = 2 * (precision_under * recall_under) / (precision_under + recall_under)

print("Random Under Sampling:")
print(f"Accuracy: {accuracy_under:.2f}")
print(f"Precision: {precision_under:.2f}")
print(f"Recall: {recall_under:.2f}")
print(f"F1-Score: {f1_score_under:.2f}")


# In[ ]:


RANDOM OVER SAMPLING


# In[133]:


from imblearn.over_sampling import RandomOverSampler
ro = RandomOverSampler()
ro_X,ro_y = ro.fit_resample(X,y)


# In[134]:


ro_y.value_counts()


# In[136]:


X_train1,X_test1,y_train1,y_test1 = train_test_split(ro_X,ro_y, test_size=0.2,random_state=21)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[138]:


LR1 = LogisticRegression()
LR1.fit(X_train1,y_train1)


# In[140]:


LR1.score(X_test1,y_test1)*100


# In[142]:


y_pred1 = LR1.predict(X_test1)
y_pred1


# In[146]:


confusion_matrix(y_test1,y_pred1)


# In[150]:


# Confusion Matrix values
TN_over, FP_over, FN_over, TP_over = 468, 247, 231, 492

# Calculations
accuracy_over = (TP_over + TN_over) / (TP_over + TN_over + FP_over + FN_over)
precision_over = TP_over / (TP_over + FP_over)
recall_over = TP_over / (TP_over + FN_over)
f1_score_over = 2 * (precision_over * recall_over) / (precision_over + recall_over)

print("Random Over Sampling:")
print(f"Accuracy: {accuracy_over:.2f}")
print(f"Precision: {precision_over:.2f}")
print(f"Recall: {recall_over:.2f}")
print(f"F1-Score: {f1_score_over:.2f}")


# In[152]:


from sklearn.metrics import confusion_matrix, classification_report

print("Classification Report:\n", classification_report(y_test1, y_pred1))


# In[ ]:




