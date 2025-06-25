# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(np.int8)

X = X / 300
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Plot some sample images
fig, axes = plt.subplots(1, 8, figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].reshape(28, 28), cmap='mako')
    ax.set_title(y_train[i])
    ax.axis('off')
plt.show()

mlp_default = MLPClassifier(random_state=1, max_iter=20)
mlp_default.fit(X_train, y_train)

y_pred_default = mlp_default.predict(X_test)

print("Accuracy (default):", accuracy_score(y_test, y_pred_default))
print(classification_report(y_test, y_pred_default))

cm = confusion_matrix(y_test, y_pred_default)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Default MLP)')
plt.show()

configs = [
    {'hidden_layer_sizes': (50,), 'activation': 'relu'},
    {'hidden_layer_sizes': (100,), 'activation': 'relu'},
    {'hidden_layer_sizes': (100, 50), 'activation': 'tanh'},
    {'hidden_layer_sizes': (50, 50, 50), 'activation': 'relu'},
]


results = []

for cfg in configs:
    print(f"Training: {cfg}")

    # Train NN
    clf = MLPClassifier(max_iter=20, random_state=1, **cfg)
    clf.fit(X_train, y_train)

    # Evaluate NN
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Configuration: {cfg}, Accuracy: {acc:.4f}")
    results.append({'config': cfg, 'accuracy': acc})
    print(f"Accuracy: {acc:.4f}")


