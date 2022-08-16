# The data included in this project is from the Kaggle website, and included a variety of different categories
# relating to credit risk and probability of default.

# We are aiming to predict risk of default, as specified by the heading "SeriousDeliqin2yrs". We will use a few
# different types of machine learning techniques algorithms to predict risk of credit default, and identify the most
# accurate technique.


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


# ----------------------------------------------------------------------------------------------------------------------

#  PART 1: DATA PREPROCESSING, INSPECTION, AND FEATURE SELECTION
# Two sets of functions: 1) Data preprocessing 2) Feature selection and scoring
# These functions will act to prepare all the data before application of machine learning models:


# Create a function to do all of the data cleaning, with the required data frame as input:
def data_clean(input_csv):
    # Import the required dataset:
    data = pd.read_csv(input_csv)

    # Deal with any Nans --> replace with zeros:
    data = data.fillna(0)

    # Our target variable is credit risk default, referred to here as 'SeriousDlqin2yrs'.
    defaulted = data.pop("SeriousDlqin2yrs")  # Isolate solar output as an independent variable
    input_data = data  # Rename

    # Remove useless parameters
    feature_output = input_data.drop(['Index'], axis=1)

    return feature_output, defaulted


# Create a function to identify feature importance:
def feature_selection(features, target):
    # Feature extraction
    selector = SelectKBest(score_func=f_classif, k='all').fit(features, target)
    scores = selector.scores_  # We now have a series of scores for each feature

    # Feature names:
    feature_names = list(features)

    # Order the variables by feature importance:
    feature_imp = pd.Series(scores, index=feature_names).sort_values(ascending=False)

    # Extract only the most important variables (n = 7):
    selected_features = (feature_imp.nlargest(7))

    return feature_imp


# Apply the preprocessing function to the data file:
data_output = data_clean("Credit_risk_modelling/cs-training.csv")
df_variables = data_output[0]  # Predictor variables
default = data_output[1]  # Target variable: default risk

# Visualize feature importance by applying the feature selection function:
feature_vars = feature_selection(df_variables, default)

# Create a new dataframe with only the most important features, as indicated by the feature selection function. This
# part can be manipulated to include more or less features.

df_var_selected = df_variables[["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate", "age",
                                "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
                                "NumberOfOpenCreditLinesAndLoans",
                                "MonthlyIncome", "DebtRatio"]]


# Now we have a workable dataframe to build machine learning models from!

# ----------------------------------------------------------------------------------------------------------------------

