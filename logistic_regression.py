import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_test_data(df):
    df = df.dropna(subset=['total_bedrooms'])

    Y = df[["ocean_proximity"]]

    X = df.drop(columns=['total_bedrooms', 'ocean_proximity'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

def predict(X, w, b):
    probs = softmax(X, w, b)
    return np.argmax(probs, axis = 1)

def accuracy(X, Y, w, b):
    Y_true = np.argmax(Y, axis=1)
    Y_pred = predict(X, w, b)
    return np.mean(Y_true == Y_pred)

def softmax(X, w, b):
    logits = np.dot(X, w.T) + b
    logits -= np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy(X, Y, w, b, epsilon=1e-12):
    Y_pred = softmax(X, w, b)
    Y_pred = np.clip(Y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(Y * np.log(Y_pred), axis=1)
    return np.mean(loss)

def gradients(X, Y, w, b):
    m = X.shape[0]
    Y_pred = softmax(X, w, b)
    error = Y_pred - Y
    dw = np.dot(error.T, X) / m
    db = np.mean(error, axis=0)
    return dw, db

def logistic_regression(X, Y, alpha, iterations):
    n_samples, n_features = X.shape
    n_classes = Y.shape[1]

    w = np.zeros((n_classes, n_features))
    b = np.zeros(n_classes)

    for i in range(iterations):
        dw, db = gradients(X, Y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            cost = cross_entropy(X, Y, w, b)
            print(f"Iteration: {i}, Cost: {cost:.4f}")
    return w, b

df = pd.read_csv("./housing.csv")
train_x, val_x, train_y, val_y = train_test_data(df)
train_y_one_hot = pd.get_dummies(train_y).values
val_y_one_hot = pd.get_dummies(val_y).values

w, b = logistic_regression(train_x, train_y_one_hot, alpha=0.5, iterations=10000)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
model.fit(train_x, train_y)

print(f"My model: {accuracy(val_x, val_y_one_hot, w, b)}\nLogReg: {model.score(val_x, val_y)}")

y_pred =  model.predict(val_x)


cm = confusion_matrix(val_y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(train_y))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.show()
