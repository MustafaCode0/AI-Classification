import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Reading the date
data = pd.read_csv("bank.csv", delimiter=";")
df = pd.DataFrame(data)

# checking what is there in the dataset
print(df.info())

df2 = df[["y", "job", "marital", "default", "housing", "poutcome"]]

# creating df3
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])
# adding y to the heatmap so i can observe the correlation.
df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

# setting fig size so data not messed up and can be read
plt.figure(figsize=(11, 9))
sns.heatmap(df3.corr(numeric_only=True).round(2), annot=True, cmap="coolwarm", annot_kws={"size": 8})
plt.tight_layout()
plt.show()

"""
after obseving the heatmap I see very little correlation between the variables and the target y. the strongests are poutcome_success 0.28
which means if customer before said yes to the bank for previous offer, would likely say yes for the deposit as well. the second is poutcome_unknown
which is -0.16 which says if customer is new never been contacted is more likey to say no.
"""

# 5 selecting y from df3 and setting the rest of columns to x
y = df3["y"]
x = df3.drop(columns=["y"])

# splitting the data 75/25
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

# creating model and training it
model = LogisticRegression()
model.fit(x_train, y_train)

# predicting y on test data
y_pred = model.predict(x_test)

# measuring how accurate the model performed using metrics confusion matrix
# and accuracy score
con_mat = confusion_matrix(y_test, y_pred)
acc_sco = accuracy_score(y_test, y_pred)

# i'm both printing and putting the result on heatmap
print(con_mat)
print(acc_sco)
sns.heatmap(con_mat, annot=True, fmt="d")
plt.show()

# creating and training knn model
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)

# making prediction based on test data
y_pred_knn = model_knn.predict(x_test)

# measuring the results
con_mat_knn = confusion_matrix(y_test, y_pred_knn)
acc_sco_knn = accuracy_score(y_test, y_pred_knn)

# printing and visualizing them
print(con_mat_knn)
print(acc_sco_knn)
sns.heatmap(con_mat_knn, annot=True, fmt="d")
plt.show()

"""
Comparing the two models LogisticRegression and KNN:

in terms of accuracy the logistic model had 0.897 accuracy score which is higher than knn
which was 0.873. measurement with confusion matrix shows that logistic model predicted true negatives "no" 996 times and
knn predicted 977 times, logistic model miss predicted false postives "yes" 9 times while knn miss predicted 28 times
which shows that logistic model was more accurate as it predicted more corrects and miss less than knn model.
i noticed that also the dataset has a lot of "no" which makes it imbalanced and affects the model predictions.
"""
