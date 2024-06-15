#!/usr/bin/env python
# coding: utf-8

# # Mushroom Data
# 
# This study explores the application of machine learning techniques, specifically Support Vector Machines (SVM), in the classification of mushroom species based on various descriptive features. The dataset consists of attributes such as cap shape, odor, and habitat, encoded and analyzed to distinguish edible from poisonous mushrooms.
# 
# # Introduction
# 
# The analysis begins with preprocessing steps, including data loading, categorical encoding, and feature scaling. Exploratory Data Analysis (EDA) techniques are employed to understand the dataset's distribution and characteristics. Initial model evaluation is conducted using cross-validation to estimate baseline performance.
# The study progresses with hyperparameter tuning of the SVM model, focusing on parameters such as `C` and `gamma` to optimize classification accuracy. Results of the tuned SVM model are presented, showcasing its effectiveness in accurately predicting mushroom types on a held-out test set.
# Additionally, a synthetic data example demonstrates SVM's capability to delineate decision boundaries in a two-dimensional feature space, illustrating the model's interpretability and generalization.
# Furthermore, leveraging linear regression, the study identifies influential data points and refines the model to improve overall predictive performance. The cleaned dataset and finalized regression model exhibit robustness, as evidenced by a high adjusted R-squared.
# This comprehensive approach not only validates the SVM model's efficacy in mushroom classification but also highlights the interpretive power of machine learning in biological data analysis.

# Part 1: Mushroom Data Analysis and SVM Model Selection

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mushroom_data = pd.read_csv('mushrooms.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
for column in mushroom_data.columns:
    mushroom_data[column] = label_encoder.fit_transform(mushroom_data[column])

# Separate features and target variable
X = mushroom_data.drop('class', axis=1)
y = mushroom_data['class']
print(mushroom_data.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 2: Initial Model Evaluation with Cross-Validation

# In[2]:


# Initialize SVM classifier with RBF kernel
nlsvm = SVC(kernel='rbf', random_state=42)

# Perform cross-validation to estimate model performance
scores = cross_val_score(nlsvm, X, y, cv=5)
print("Cross-Validation Mean Accuracy: {:.3f}".format(np.mean(scores)))


# Step 3: Hyperparameter Tuning (C Parameter)

# In[3]:


param_grid_C = {'C': [0.1, 1, 10, 100]}
grid_search_C = GridSearchCV(nlsvm, param_grid_C, cv=5)
grid_search_C.fit(X_train_scaled, y_train)

# Print best C parameter and its cross-validation score
print("Best C parameter:", grid_search_C.best_params_)
print("Best cross-validation score with best C parameter: {:.3f}".format(grid_search_C.best_score_))

# Use the best estimator for further analysis
nlsvm_C_best = grid_search_C.best_estimator_


# Step 4: Hyperparameter Tuning (Gamma Parameter)

# In[4]:


# Hyperparameter tuning for gamma parameter
param_grid_gamma = {'gamma': [0.001, 0.01, 0.1, 1, 10]}
grid_search_gamma = GridSearchCV(nlsvm, param_grid_gamma, cv=5)
grid_search_gamma.fit(X_train_scaled, y_train)

# Print best gamma parameter and its cross-validation score
print("Best gamma parameter:", grid_search_gamma.best_params_)
print("Best cross-validation score with best gamma parameter: {:.3f}".format(grid_search_gamma.best_score_))

# Use the best estimator for further analysis
nlsvm_gamma_best = grid_search_gamma.best_estimator_


# Step 5: Model Evaluation and Visualization

# In[6]:


# Evaluate the best model on the test set
y_pred = nlsvm_C_best.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on test set: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', annot_kws={'size': 16})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# ### Part 2: Nonlinear SVM
# 

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to generate and split synthetic data
def prepare_data():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    return X, y

# Function to scale and split data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# Function to train and plot SVM decision boundary
def train_and_plot_svm(X_train_scaled, y_train):
    svm_clf = SVC(kernel='linear', random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.Paired, s=30)
    
    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    Z = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k')
    
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Main workflow for synthetic data example
if __name__ == "__main__":
    X, y = prepare_data()
    X_train_scaled, X_test_scaled, y_train, y_test = split_data(X, y)  # Ensure correct unpacking of y_train and y_test
    train_and_plot_svm(X_train_scaled, y_train)


# In[8]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming mushroom_data is already loaded and preprocessed

# Fit a multiple linear regression model
X = sm.add_constant(X)  # Add a constant (intercept) to the model
# Determine the column names based on the number of features + 1 (for the constant)
column_names = ['const'] + [f'feature{i+1}' for i in range(X.shape[1] - 1)]
X = pd.DataFrame(X, columns=column_names)  # Convert X to DataFrame with appropriate column names
y = pd.Series(y, name='target')  # Convert y to Series with appropriate name

model_multi = sm.OLS(y, X).fit()

# Calculate leverage
leverage = model_multi.get_influence().hat_matrix_diag

# Identify observations with leverage > 0.4
high_leverage_indices = list(np.where(leverage > 0.4)[0])

# Calculate squared residuals
squared_residuals = model_multi.resid ** 2

# Plot leverage vs. squared residuals
plt.figure(figsize=(10, 6))
plt.scatter(leverage, squared_residuals, alpha=0.5)
plt.scatter(leverage[high_leverage_indices], squared_residuals[high_leverage_indices], color='red', alpha=0.5, label='High Leverage')
plt.title('Leverage vs. Squared Residuals')
plt.xlabel('Leverage')
plt.ylabel('Squared Residuals')
plt.legend()
plt.show()

# Print indices of high-leverage points
print("Indices of high-leverage points:", high_leverage_indices)

# Exclude high-leverage points to create a cleaned dataset
X_clean = X.drop(high_leverage_indices)
y_clean = y.drop(high_leverage_indices)

# Fit final model on cleaned dataset
model_final = sm.OLS(y_clean, X_clean).fit()

# Check adjusted R-squared of the final model
if model_final.rsquared_adj > 0.95:
    print(f"Final model R-squared (adjusted): {model_final.rsquared_adj:.3f}")
else:
    print("Final model does not meet R-squared > 0.95 criterion.")

# Print summary of the final model
print(model_final.summary())

