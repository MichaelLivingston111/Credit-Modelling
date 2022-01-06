
# Import:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from scipy.io import arff


# The dataset contains information on bankruptcy rates in Polish companies. The data was collected from Emerging Markets
# Information Service (EMIS, [Web Link]), which is a database containing information on emerging markets around the
# world. The bankrupt companies were analyzed through 2000-2012, while the still operating companies were
# evaluated from 2007 to 2013.


# Load the dataset from the 5th year of the data collection:
data = arff.loadarff('Credit_risk_modelling/Polish_Business_Bankruptcy/5year.arff')
df = pd.DataFrame(data[0])  # Select all the data


# The dataframe contains 64 different 'attributes', along with a categorical values for either bankruptcy
# or non-bankruptcy. Example of attributes include: net profit / total assets, total liabilities / total assets,
# working capital / total assets etc etc....

# For simplicity sake I will not rename the columns. For reference, each financial attribute column can be easily
# accessed using this link: http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data#


# Remove the bankruptcy data from the above dataframes and store it as a target variable:
y = df.pop('class')
y = pd.to_numeric(y)  # 0 = no bankruptcy, 1 = bankruptcy


# DATA CLEANING:
# Analysis of null information columns, if any:
df.isnull().sum()  # Several columns have lots of Nan values.

# Need to deal with Nans - Imputing with MICE (multiple imputation by chained equations).  This algorithm is applied
# to every column that has some missing values and fits a linear regression with the present values. After that,
# it uses these linear functions to impute the NaN values.

lr = LinearRegression()
imp = IterativeImputer(estimator=lr, missing_values=np.nan, max_iter=10, verbose=2,
                       imputation_order='roman', random_state=0)

df_transform = imp.fit_transform(df)
df_transform = pd.DataFrame(df_transform)  # Convert to a dataframe
df_transform.isnull().sum()  # No longer any columns with Nans! They have all been replaced with predicted values.


# STANDARDIZATION: Scale all columns between -1 and 1:
ss = StandardScaler()

df_scaled = ss.fit_transform(df_transform)
df_scaled = pd.DataFrame(df_scaled)  # Convert to a dataframe


# Convert all columns types in to integers:
df_scaled.dtypes  # All are either int64 or float64, so no need to convert categorical data to numeric.
print(df_scaled.head())

# Rename the training set to X:
X = df_scaled


# FEATURE SELECTION: We need to filter out any junk variables that may be unrelated ot our target variable (
# bankruptcy) in order to improve the accuracy of the model downstream. This abides by the principal of "Garbage in
# garbage out".

X_new = SelectKBest(k=48).fit_transform(df_scaled, y)  # Create the selection model based off the F statistic in ANOVA
X_new = pd.DataFrame(X_new)  # Create new dataframe


# Split the data into training and testing sets:
x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)

print("Train data has {} data points, test data has {} data points" .format(x_train.shape[0], x_test.shape[0]))


# CREATE THE NEURAL NETWORK MODEL:
model = models.Sequential()
model.add(Dense(48, input_dim=48, activation='relu'))  # input dimensions
model.add(Dense(24, activation='relu'))  # Dense, fully connected layers
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation function for values 0-1

# Compile the model:
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model with the training sets:
history = model.fit(x_train, y_train, validation_split=0.3, epochs=60,
                    batch_size=600, validation_data=(x_test, y_test))


# Evaluating the model: determine how well the model performed by looking at its performance on the test data:
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_acc)

predictions = model.predict(x_test)
predictions.max()

# All companies with a bankruptcy probability >30% are considered as a high probability and assigned a value of 1.
predictions_binary = np.where(predictions > 0.3, 1, 0)


# Create a confusion matrix to estimate the models accuracy:
cnf_matrix = metrics.confusion_matrix(y_test, predictions_binary)

acc_score = accuracy_score(y_test, predictions_binary)
print(acc_score*100)

# Create a heatmap of the confusion matrix to quantify the prediction accuracy:
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True Default')
plt.xlabel('Predicted Default')

# Print the accuracy metrics.
print("Accuracy:", metrics.accuracy_score(y_test, predictions_binary))
print("Precision:", metrics.precision_score(y_test, predictions_binary))
print("Recall:", metrics.recall_score(y_test, predictions_binary))


# Visualize the accuracy and loss function of the model over each epoch. This plots will also help to identify if the
# model is over-fitting on the training data.

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# CONCLUSION: This neural network forecasts the probability of bankruptcy (in one year) for 5910 companies with an
# accuracy of 92%. It can accurately predict future bankruptcy with a recall precision (true positive) of 35 - 40%. It
# predicts companies that will not undergo bankruptcy (true negative) with a precision of >97%. Importantly,
# this model does not seem to be over-fitting the data and therefore may suitable for application on new datasets and
# financial statements.
