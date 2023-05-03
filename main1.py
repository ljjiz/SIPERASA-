import streamlit as st

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


st.title('Sistem Perbandingan Akademik Siswa')

st.write("""
# Membandingkan Algoritma DecisionTree dengan RandomForest
Manakah yang terbaik ?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Kaggle', '')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('DecisionTree', 'Random Forest')
)

def get_dataset(name):
    data = None
    if name == 'Kaggle':
        data = pd.read_csv("S:\Indonesia\semester 5\Big Data\excel\student-mat.csv")
    else:
        data = pd.read_csv("S:\Indonesia\semester 5\Big Data\excel\student-mat.csv")
    


   
data = pd.read_csv("S:\Indonesia\semester 5\Big Data\excel\student-mat.csv")


st.write('Shape of dataset:', data.shape)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'DecisionTree':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        
    else:
        max_depth = st.sidebar.slider('max_depth', 3, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'DecisionTree':
        clf = clf = DecisionTreeClassifier(
            max_depth=params['max_depth'],random_state=42)
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
data["activities"] = lb.fit_transform(data["activities"])
categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

# Define the categorical target variable
bins = [0, 10, 15, 20]
labels = ['low', 'medium', 'high']
data['target_variable'] = pd.cut(data['AVG'], bins=bins, labels=labels)

# Set the target variabl  e

y = data["target_variable"]

# Split the data into training and testing sets using decision tree algorithm
X_train, X_test, y_train, y_test = train_test_split(data.drop(["AVG", "target_variable"], axis=1), y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set and evaluate performance using accuracy score
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)


# # #### PLOT DATASET ####
# # # Project the data onto the 2 primary principal components
# pca = PCA(2)
# X_projected = pca.fit_transform(X)

# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]

# fig = plt.figure()
# plt.scatter(x1, x2,
#         c=y, alpha=0.8,   
#         cmap='viridis')

# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# # plt.colorbar()

# # #plt.show()
# st.pyplot(fig)
