import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# reading data
df = pd.read_csv("weight-height.csv")

# setting x and y values
x = df[["Height"]]
y = df["Weight"]

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# creating scaler and normalizing train and test data
mm = MinMaxScaler()
x_train_norm = mm.fit_transform(x_train)
x_test_norm = mm.transform(x_test)

# creating std and standardizing train and test data with it
std = StandardScaler()
x_train_std = std.fit_transform(x_train)
x_test_std = std.transform(x_test)

# creating knn model and setting neighboring to 5, and predicting y
knn_unscaled = KNeighborsRegressor(n_neighbors=5)
knn_unscaled.fit(x_train, y_train)
unscaled_y_pred = knn_unscaled.predict(x_test)

# measuring and printing r2 scores
print("Unscaled r2: ", r2_score(y_test, unscaled_y_pred))

# training model with x_train_norm and repeating above steps for normalization data
knn_norm = KNeighborsRegressor(n_neighbors=5)
knn_norm.fit(x_train_norm, y_train)
norm_y_pred = knn_norm.predict(x_test_norm)

# measuring and printing r2 scores for normalization model
print("Normalization r2: ", r2_score(y_test, norm_y_pred))

# training model with x_train_norm and repeating above steps for standardization data
knn_std = KNeighborsRegressor(n_neighbors=5)
knn_std.fit(x_train_std, y_train)
std_y_pred = knn_std.predict(x_test_std)

# measuring and printing r2 scores for standardization model
print("Standardization r2: ", r2_score(y_test, std_y_pred))

"""
Thoughts and Conclusion:

After running this code we get 
Unscaled r2:  0.8197710760798875
Normalization r2:  0.8197710760798875
Standardization r2:  0.8197710760798875

they are identical numbers and that's because scaling data set with one column
does not affect the result. The tallest person remains the closest neighbor whether their height
is measured in centimeters or a 0 to 1 scale or standard deviations.
"""
