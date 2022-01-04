
# Imports:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

# We are aiming to predict risk of default, as specified by the heading "SeriousDeliqin2yrs"

# Read the data set:
data = pd.read_csv("Credit_risk_modelling/cs-training.csv")


# Isolate relevant columns:
data = data[["SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]]

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
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=0)

print("Train data has {} data points, test data has {} data points" .format(x_train.shape[0], x_test.shape[0]))

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# New we can fit and train the model with Sklearn
model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)
model.fit(x_train, y_train)

# Make predictions on th entire test data set:
y_pred = model.predict(x_test)

# Create a confusion matrix to estimate the models accuracy:
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

acc_score = accuracy_score(y_test, y_pred)
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





