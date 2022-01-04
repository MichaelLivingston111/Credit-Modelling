
# Imports:
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

# We are aiming to predict risk of default, as specified by the heading "SeriousDeliqin2yrs"

# Read the data set:
data = pd.read_csv("Credit_risk_modelling/cs-training.csv")


# Isolate relevant columns:
full_data = data[["SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]]

data = data[["SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
             "NumberOfTimes90DaysLate", "MonthlyIncome"]]

feature_names = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]

# Analysis of null information columns, if any:
data.isnull().sum()  # Clearly two columns have null values.

# Replace the NaNs with zeros:
data = data.fillna(0)


# Convert all columns types in to integers:
data.dtypes  # All are either int64 or float64
print(data.head())

# Create correlation values between columns:
corr = data.corr().abs()

# Visualize the relationships between all variables:
figure = plt.figure(figsize=(8, 8))
plt.matshow(data.corr(), fignum=figure.number)
plt.xticks(range(data.shape[1]), data.columns, rotation=90)
plt.yticks(range(data.shape[1]), data.columns)
cb = plt.colorbar()

# Isolate the target variable:
Y = data["SeriousDlqin2yrs"]
X = data.drop("SeriousDlqin2yrs", axis=1)


# Split the data into traing and tetsing sets:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

print("Train data has {} data points, test data has {} data points" .format(x_train.shape[0], x_test.shape[0]))

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# New we can fit and train the model(s) with Sklearn.

# Logistic Regression:
model_LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)
model_LR.fit(x_train, y_train)


# K-nearest neighbors:
model_KN = KNeighborsClassifier(n_neighbors=1)
model_KN.fit(x_train, y_train)


# Random forest:
model_RF = RandomForestClassifier(n_estimators=200)
model_RF.fit(x_train, y_train)

# Here, we can deduce which features are the most important and drop the remaining ones:
feature_imp = pd.Series(model_RF.feature_importances_, index=feature_names).sort_values(ascending=False)
feature_imp

sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# After taking this into account, I created another dataframe at the top of this file with the most important features.


# Make predictions on the entire test data set:
y_pred_LR = model_LR.predict(x_test)
y_pred_KN = model_KN.predict(x_test)
y_pred_RF = model_RF.predict(x_test)

# Create a confusion matrix to estimate the models accuracy:
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_LR)
cnf_matrix_KN = metrics.confusion_matrix(y_test, y_pred_KN)
cnf_matrix_RF = metrics.confusion_matrix(y_test, y_pred_RF)


#####

# Logistic regression accuracy:
acc_score = accuracy_score(y_test, y_pred_LR)
print(acc_score*100)

# create heatmap of the confusion matrix to quantify the prediction accuracy:
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

print("Accuracy:", metrics.accuracy_score(y_test, y_pred_LR))
print("Precision:", metrics.precision_score(y_test, y_pred_LR))
print("Recall:", metrics.recall_score(y_test, y_pred_LR))


#####

# K-nearest neighbours accuracy:
acc_score = accuracy_score(y_test, y_pred_KN)
print(acc_score*100)

# create heatmap of the confusion matrix to quantify the prediction accuracy:
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix_KN), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True Default')
plt.xlabel('Predicted Default')

print("Accuracy:", metrics.accuracy_score(y_test, y_pred_KN))
print("Precision:", metrics.precision_score(y_test, y_pred_KN))
print("Recall:", metrics.recall_score(y_test, y_pred_KN))

#####

# Random forest accuracy:
acc_score = accuracy_score(y_test, y_pred_RF)
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

print("Accuracy:", metrics.accuracy_score(y_test, y_pred_RF))
print("Precision:", metrics.precision_score(y_test, y_pred_RF))
print("Recall:", metrics.recall_score(y_test, y_pred_RF))


# From this we can deduce that while all the models are fairly accurate, they are not great at predicting actual
# defaults. Instead, they are good at predicting people who will not default. The logistic regression is especially
# bad in this sense. The other two models are significantly better, however there is still room for much improvement.
# We may be limited by the data we have.






