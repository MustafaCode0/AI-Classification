import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# reading the data and putting it in dataframe
df = pd.read_csv("data_banknote_authentication.csv")

# setting x value to all columns excluding class column and y to class
x = df.drop(columns="class")
y = df["class"]

# splitting the data 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# making, training the model and predicting the y
svc_model = SVC(kernel="linear")
svc_model.fit(x_train, y_train)
svc_y_pred = svc_model.predict(x_test)

# checking the model performance
print(confusion_matrix(y_test, svc_y_pred))
print(classification_report(y_test, svc_y_pred))

# making, training the model and predicting the y
rbf_model = SVC(kernel="rbf")
rbf_model.fit(x_train, y_train)
rbf_y_pred = rbf_model.predict(x_test)

# printing the model performance result
print(confusion_matrix(y_test, rbf_y_pred))
print(classification_report(y_test, rbf_y_pred))

"""
svc linear kernel model performed very well, looking at confusion matrix result it got
152 TN and 121TP and only 2FN which is good. also it got 0.99 recall which is also 
very high. On the other hand svc kernel rbf did even better performance it got 100%
accuracy and predicted 100% correctly also in confusion matrix, and this tells us that 
the dataset has non_linear relationships and kernel rbf handles it better than linear. 
"""
