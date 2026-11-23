# 9A Work

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier


x1 = 6
x2 = 1 

w1 = -0.5
w2 = -0.8
b = 4

z = w1 * x1 + w2 * x2 + b

sigmoid = lambda value: 1 / (1 + np.exp(-value))
prediction = sigmoid(z)


z_values = np.linspace(-6, 6, 300)
sigmoid_values = 1 / (1 + np.exp(-z_values))
tanh_values = np.tanh(z_values)
relu_values = np.maximum(0, z_values)


moons = pd.read_csv("C:\\Users\\abadj\\Documents\\GitHub\\CSC1171\\Week 1\\week9-DataIntro\\data\\week9_two_moons.csv")
feature_names = ["feature_1", "feature_2"]

X = moons[feature_names].to_numpy()
y = moons["target"].to_numpy()

moons.head()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

def plot_decision_boundary(model, X_data, y_data, title):
    """Plot a two-dimensional classifier's decision regions and observations."""
    x_min, x_max = X_data[:, 0].min() - 0.6, X_data[:, 0].max() + 0.6
    y_min, y_max = X_data[:, 1].min() - 0.6, X_data[:, 1].max() + 0.6
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    predicted_classes = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, predicted_classes, alpha=0.2, levels=[-0.5, 0.5, 1.5])
    plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, edgecolor="black", linewidth=0.3, alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.grid(alpha=0.15)
    plt.show()


plot_decision_boundary(
    logistic_model,
    X_test,
    y_test,
    "Logistic regression: test observations and decision boundary"
)

neural_network = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(12,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.01,
        max_iter=2000,
        random_state=42
    ))
])

neural_network.fit(X_train, y_train)
mlp_predictions = neural_network.predict(X_test)

plot_decision_boundary(
    neural_network,
    X_test,
    y_test,
    "MLP: test observations and decision boundary"
)

ConfusionMatrixDisplay.from_predictions(y_test, mlp_predictions)
plt.title("MLP confusion matrix")
plt.show()

comparison = pd.DataFrame({
    "model": ["Logistic regression", "MLP"],
    "test_accuracy": [
        accuracy_score(y_test, logistic_predictions),
        accuracy_score(y_test, mlp_predictions)
    ]
})

comparison

mlp_es = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(12,), learning_rate_init=0.01, max_iter=2000,
        early_stopping=True, validation_fraction=0.1, random_state=42
    ))
])
mlp_es.fit(X_train, y_train)
m = mlp_es.named_steps["model"]

print(f"Epochs: {m.n_iter_}")
print(f"Best validation accuracy: {m.best_validation_score_:.3f}")
print(f"Test accuracy: {mlp_es.score(X_test, y_test):.3f}")