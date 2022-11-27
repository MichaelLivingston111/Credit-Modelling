
# The data included in this project is from the Kaggle website, and included a variety of different categories
# relating to credit risk and probability of default.

# We are aiming to predict risk of default. We will use a few
# different types of machine learning techniques algorithms to predict risk of credit default, and identify the most
# accurate techniques. This varies from the previous model in terms of input parameters.

# What is required to use these algorithms: Number of days past due payment (30-59, 60-89, 90+ days), age,
# number of dependants, number of open credit lines, monthly income and debt ratio.


# IMPORTS:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestRegressor

import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# ----------------------------------------------------------------------------------------------------------------------

#  PART 1: DATA PREPROCESSING, INSPECTION, AND FEATURE SELECTION
# Two sets of functions: 1) Data preprocessing 2) Feature selection and scoring
# These functions will act to prepare all the data before application of machine learning models:


# Create a function to do all of the data cleaning, with the required data frame as input:
def data_clean(input_csv):

    # Import the required dataset:
    data = pd.read_csv(input_csv)

    # Deal with any Nans --> replace with zeros:
    data = data.dropna(axis=0)

    # Remove any unrealistic/errors in age or employment history:
    data.drop(data[data['person_emp_length'] >= 90].index, inplace=True)  # employment length
    data.drop(data[data['person_age'] >= 90].index, inplace=True)  # employment length

    # Our target variable is credit risk default, referred to here as 'loan_status' (1 = default).
    defaulted = data.pop("loan_status")  # Isolate solar output as an independent variable
    feature_output = data  # Rename

    # identify all categorical variables
    cat_columns = feature_output.select_dtypes(['object']).columns

    # convert all categorical variables to numeric
    feature_output[cat_columns] = feature_output[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # Now we need to split into training and testing sets:
    x_train, x_test, y_train, y_test = train_test_split(feature_output, defaulted, test_size=0.1, random_state=0)

    return x_train, x_test, y_train, y_test


alldata = data_clean("credit_risk_dataset.csv")

feature_train = alldata[0]
feature_test = alldata[1]
default_train = alldata[2]
default_test = alldata[3]

feature_train.max(axis='rows')
feature_train.dtypes

# Create a function to identify and select feature based off order of importance:
def feature_selection(features, target):
    # Feature extraction
    selector = SelectKBest(score_func=f_classif, k='all').fit(features, target)
    scores = selector.scores_  # We now have a series of scores for each feature

    # Feature names:
    feature_names = list(features)

    # Order the variables by feature importance:
    feature_imp = pd.Series(scores, index=feature_names).sort_values(ascending=False)

    # Extract only the most important variables (n = 7; top 7 most important features):
    selected_features = (feature_imp.nlargest(7))

    # Convert the index (parameters) to a list:
    selected_features = selected_features.index.values.tolist()

    # Use the selected feature list to extract the selected features from the original dataframe:
    selected_features_df = features.filter(items=selected_features)

    return selected_features_df
