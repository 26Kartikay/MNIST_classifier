import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (scaling pixel values between 0 and 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Flatten images into 1D arrays
X_train_flattened = X_train.reshape(len(X_train), 28 * 28)
X_test_flattened = X_test.reshape(len(X_test), 28 * 28)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation="relu"),
    keras.layers.Dense(10, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_flattened, y_train, epochs=5)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test_flattened, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Predictions
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Plot confusion matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Display an example test image and its prediction
plt.matshow(X_test[0])
plt.title(f"Predicted Label: {y_predicted_labels[0]}")
plt.show()
