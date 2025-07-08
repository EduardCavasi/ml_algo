import pandas as pd
import matplotlib.pyplot as plt
from pyexpat import model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

def train_test_data(df: pd.DataFrame):
    Y = df.median_house_value
    X = df.drop(columns=['median_house_value', 'total_bedrooms', 'ocean_proximity'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

def compute_cost_squared_error(X, Y, w, b):
    m = X.shape[0]
    Y_pred = np.dot(X, w) + b
    cost = np.sum((Y_pred - Y)**2) / (2 * m)
    return cost

def compute_cost_mae(X, Y, w, b):
    m = X.shape[0]
    Y_pred = np.dot(X, w) + b
    cost = np.sum(abs(Y_pred - Y)) / (2 * m)
    return cost

def compute_gradients(X, Y, w, b):
    m = X.shape[0]
    dw = np.dot(X.T, (np.dot(X, w) + b - Y)) / m
    db = np.sum((np.dot(X, w) + b) - Y) / m
    return dw, db

def linear_regression(X, Y, alpha, iterations):
    w = np.zeros(X.shape[1])
    b = 0
    for i in range(iterations):
        dw, db = compute_gradients(X, Y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            print(f"Iteration: {i}, Cost: {compute_cost_squared_error(X, Y, w, b)}")
    return w, b

def predict(X, w, b):
    return np.dot(X, w) + b

if __name__ == "__main__":
    df = pd.read_csv("./housing.csv")

    train_x, val_x, train_y, val_y = train_test_data(df)
    #print(train_x, val_x, train_y, val_y)

    #pd.set_option('display.max_columns', None)
    #print(df.describe())
    #print(df.columns)

    #plt.scatter(df.total_bedrooms, df.median_house_value)
    #plt.xlabel("Total Bedrooms")
    #plt.ylabel("Price")
    #plt.show()

    w, b = linear_regression(train_x, train_y, 0.1, 1000)
    #print(w)
    #print(b)

    compare_model = SGDRegressor()
    compare_model.fit(train_x, train_y)
    #print(compare_model.coef_)
    #print(compare_model.intercept_)

    compare_model_2 = LinearRegression()
    compare_model_2.fit(train_x, train_y)

    print(f"My model: {compute_cost_squared_error(val_x, val_y, w, b)}\nSGDRegressor: {compute_cost_squared_error(val_x, val_y, compare_model.coef_, compare_model.intercept_)}\nLinearRegression: {compute_cost_squared_error(val_x, val_y, compare_model_2.coef_, compare_model_2.intercept_)}")
    print(f"My model: {compute_cost_mae(val_x, val_y, w, b)}\nSGDRegressor: {compute_cost_mae(val_x, val_y, compare_model.coef_, compare_model.intercept_)}\nLinearRegression: {compute_cost_mae(val_x, val_y, compare_model_2.coef_, compare_model_2.intercept_)}")

    print(predict(val_x, w, b)[:5])
    print(compare_model.predict(val_x)[:5])
    print(np.array(val_y)[:5])

