# Import Required Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and Preprocess the Breast Cancer Dataset

# Load the dataset
data = load_breast_cancer()
X = data.data  # Features: 30 numeric features of cells
y = data.target  # Labels: 0 for malignant, 1 for benign

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features: neural networks perform better when features are on the same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Build the Neural Network Model

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer (30 features)
    Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    Dense(1, activation='sigmoid')  # Output layer with a sigmoid activation for binary classification
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Since it's binary classification
              metrics=['accuracy'])

# 4. Train the Model
history = model.fit(X_train, y_train, epochs=1000, batch_size=16, validation_data=(X_test, y_test))

# 5. Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# 6. Plot Training and Validation Loss/Accuracy
# Plot the accuracy curve
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 7. Make Predictions (Optional)
# Predict on the test set
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype('int32')  # Convert to binary labels (0 or 1)

# Display some of the predicted results vs true labels
for i in range(10):
    print(f"Predicted: {'Benign' if predicted_classes[i][0] == 1 else 'Malignant'}, "
          f"True: {'Benign' if y_test[i] == 1 else 'Malignant'}")

