# The data included in this project is from the Kaggle website, and included a variety of different categories
# relating to credit risk and probability of default.

# We are aiming to predict risk of default, as specified by the heading "SeriousDeliqin2yrs". We will use a few
# different types of machine learning techniques algorithms to predict risk of credit default, and identify the most
# accurate technique.

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

# TESTING FOR GIT COMMIT!!!!!!!!


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


# Apply the preprocessing function to the data file:
data_output = data_clean("Credit_risk_modelling/cs-training.csv")
df_variables = data_output[0]  # Predictor variables
default = data_output[1]  # Target variable: default risk

# Extract a dataframe with the selected feature variables using the feature_selection function::
feature_vars = feature_selection(df_variables, default)


# Now we have a workable dataframe to build machine learning models from!

# ----------------------------------------------------------------------------------------------------------------------


# Need to split the data into training and testing sets to build and test the model:
x_train1, x_test1, y_train1, y_test1 = train_test_split(feature_vars, default, test_size=0.2, random_state=0)
print("Train data has {} data points, test data has {} data points".format(x_train1.shape[0], x_test1.shape[0]))


# Build a function that creates a deep neural network, with the training sets as input:

def neural_network(xtrain, ytrain, xtest, ytest, variables, activation_fn_hidden, activation_fn_output, learning_rate,
                   loss_metric, num_epochs, batch_size):

    # CREATE THE NEURAL NETWORK:
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(variables))

    # Model architecture:
    model = keras.Sequential([
        normalizer,
        layers.Dense(8, input_dim=xtrain.shape[1], activation=activation_fn_hidden),
        layers.Dropout(0.2),
        layers.Dense(4, activation=activation_fn_hidden),
        layers.Dense(1, activation=activation_fn_output),
        layers.Dense(1)
    ])

    # Model compilation:
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_metric,
        metrics=['accuracy']
    )

    model.build()

    # Train the DNN:
    num_epochs = num_epochs
    batch_size = batch_size
    history_1 = model.fit(xtrain, ytrain, epochs=num_epochs, validation_split=0.2)  # Fitting

    loss1, mae1 = model.evaluate(xtest, ytest, verbose=2)  # Calculate model mean absolute errors and loss rates

    return model, history_1


# Build a function that creates a deep Random Forest Algorithm, with the training sets as input:

def random_forest(xtrain, ytrain, n_estimators, random_state):

    # Instantiate model with x# of decision trees
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    rf.fit(xtrain, ytrain)  # Train the RF

    return rf


# FEATURE SCALING:
sc = StandardScaler()
x_train1 = sc.fit_transform(x_train1)
x_test1 = sc.transform(x_test1)


# Apply the DNN function:
DNN_output = neural_network(x_train1, y_train1, x_test1, y_test1, feature_vars, 'relu', 'sigmoid', 0.001,
                            'binary_crossentropy', 20, 50000)

# ASSESS THE PERFORMANCE OF THE ALGORITHM: Visualizing its accuracy and loss rate over each epoch will give us
# insight into whether or not the model is over/under fitting the data:
DNN_model = DNN_output[0]  # Specify the DNN model
hist = DNN_output[1]  # Specify the loss history

# Summarize history for loss: (This is only applicable for the DNN)
acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()


# Apply the RF function:
RF_model = random_forest(x_train1, y_train1, 1000, 42)


# PREDICTIONS:

# Make predictions on the entire test data set:
y_pred_DNN = DNN_model.predict(x_test1)
y_pred_DNN_binary = np.where(y_pred_DNN > 0.3, 1, 0)  # Create binary predictions


y_pred_RF = RF_model.predict(x_test1)

# Create a confusion matrix to estimate the models accuracy:
cnf_matrix_DNN = metrics.confusion_matrix(y_test1, y_pred_DNN_binary)
cnf_matrix_RF = metrics.confusion_matrix(y_test1, y_pred_RF)





#####

# Random forest accuracy:
acc_score = accuracy_score(y_test1, y_pred_RF)
print(acc_score*100)

# Create heatmap of the confusion matrix to quantify the prediction accuracy:
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix_RF), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True Default')
plt.xlabel('Predicted Default')

print("Accuracy:", metrics.accuracy_score(y_test1, y_pred_RF))
print("Precision:", metrics.precision_score(y_test1, y_pred_RF))
print("Recall:", metrics.recall_score(y_test1, y_pred_RF))



#####

# DNN accuracy:
acc_score = accuracy_score(y_test1, y_pred_DNN_binary)
print(acc_score*100)

# Create heatmap of the confusion matrix to quantify the prediction accuracy:
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix_DNN), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True Default')
plt.xlabel('Predicted Default')

print("Accuracy:", metrics.accuracy_score(y_test1, y_pred_DNN_binary))
print("Precision:", metrics.precision_score(y_test1, y_pred_DNN_binary))
print("Recall:", metrics.recall_score(y_test1, y_pred_DNN_binary))