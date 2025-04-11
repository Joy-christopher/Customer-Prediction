#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import pandas as pd
import numpy as np

#Import Dataset
data = pd.read_csv("C:/Users/joyou/OneDrive/Desktop/Churn_Modelling.csv")


# In[2]:


#Viewing the data dimensions
data.shape

data.head()


# In[3]:


#Describing the numeric colums in the dataset
data.describe()


# In[4]:


#Viewing Datatypes 
data.info()


# In[5]:


# Checking for missing values
print(data.isnull().sum())


# In[6]:


from sklearn.model_selection import train_test_split

# Define the features and the target
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Bar chart for the 'Exited' column
sns.countplot(x='Exited', data=data)
plt.title('Distribution of Exited')
plt.show()


# In[8]:


# Identify continuous and categorical features
categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
continuous_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude the target variable from continuous features
continuous_features.remove('Exited')

print(f"Categorical features: {categorical_features}")
print(f"Continuous features: {continuous_features}")


# In[9]:


#Dropping a column
data = data.drop(['Surname'], axis=1)


# In[10]:


# Histograms for continuous features
data[continuous_features].hist(bins=15, figsize=(15, 10), layout=(4, 3))
plt.tight_layout()
plt.show()


# In[11]:


# Correlation matrix for continuous features
correlation_matrix = data[continuous_features].corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# In[12]:


# Count plots for categorical features and their exited rates
#for feature in categorical_features:
 #   plt.figure(figsize=(10, 5))
  #  sns.countplot(x=feature, hue='Exited', data=data)
   # plt.title(f'Count plot for {feature} with Exited rates')
    #plt.show()


# In[13]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the preprocessing for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, continuous_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the transformations to the training and test sets
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print(f"Preprocessed training set shape: {X_train.shape}")
print(f"Preprocessed test set shape: {X_test.shape}")


# MODEL TRAINING AND EVALUATION

# In[14]:


# Step 1: Defining the classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV


# In[15]:


#Training the models
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)


# In[16]:


#Prediction

y_pred_logreg = logreg.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_gb = gb.predict(X_test)


# In[17]:


# Function to evaluate the model and plot confusion matrix
def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, f'Confusion Matrix - {model_name}')


# In[18]:


# Function to plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()


# In[19]:


# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluate_model(y_test, y_pred_dt, "Decision Tree")


# In[ ]:





# In[20]:


# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
evaluate_model(y_test, y_pred_logreg, "Logistic Regression")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluate_model(y_test, y_pred_dt, "Decision Tree")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Support Vector Machine (SVM)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
evaluate_model(y_test, y_pred_svm, "Support Vector Machine (SVM)")

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors (KNN)")

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")


# In[ ]:




