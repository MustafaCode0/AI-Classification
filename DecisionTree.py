import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# reading the data
df = pd.read_csv("suv.csv")

# setting x and y values
x = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# splitting the data tp 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# creating standard scaler and making data standard
std = StandardScaler()
x_train_std = std.fit_transform(x_train)
x_test_std = std.transform(x_test)

# creating training and predicting with decision tree classifier with entropy
dtc = DecisionTreeClassifier(criterion="entropy", random_state=5)
dtc.fit(x_train_std, y_train)
dtc_y_pred = dtc.predict(x_test_std)

# printing confusion matrix and classifier report
print("Entropy\nConfusion matrix: ", confusion_matrix(y_test, dtc_y_pred))
print("Classification report: ", classification_report(y_test, dtc_y_pred))

# creating model with "gini" criterion
dtc_gini = DecisionTreeClassifier(criterion="gini", random_state=5)
dtc_gini.fit(x_train_std, y_train)
gini_y_pred = dtc_gini.predict(x_test_std)

# printing confufion matrix and classifier report for gini model
print("Gini\nconfusing matrix: ", confusion_matrix(y_test, gini_y_pred))
print("Classification report: ", classification_report(y_test, gini_y_pred))

"""
Entropy Model: This model achieved an overall accuracy of 85%. 
Looking at its confusion matrix, it correctly predicted 47 True Negatives (did not purchase) and 21 True Positives (purchased), 
making 12 total errors (6 false positives and 6 false negatives). Its precision and recall scores 
were balanced at 0.89 for class 0 and 0.78 for class 1.
Gini Model: This model performed better, achieving a higher overall accuracy of 88%. 
Its confusion matrix shows that it reduced classification errors to just 10 total mistakes. 
it reduced the false negatives from 6 down to 4, which boosted the recall for class 1 to 0.85 (compared to Entropy's 0.78).
"""
