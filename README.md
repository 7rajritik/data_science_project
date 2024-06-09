# hand_written_digit_prediction
# Objective:
The objective of this code is to develop a machine learning model that can accurately classify handwritten digits from the MNIST dataset.
# Data Source:
The MNIST dataset is a classic benchmark dataset in machine learning, consisting of 28x28 pixel grayscale images of handwritten digits (0 through 9). It contains a training set of 60,000 examples and a test set of 10,000 examples.

Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models

Step 2: Load and Prepare Data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

Step 3: Define Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to a 1D array
    layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons, ReLU activation
    layers.Dense(10, activation='softmax') # Output layer with 10 neurons (one for each digit), softmax activation
])

#Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
Step 4: Train the Model
model.fit(train_images, train_labels, epochs=5)

Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

Step 6: Make Predictions
predictions = model.predict(test_images)

That's it! This code will train a simple neural network model to classify handwritten digits and evaluate its performance. You can further experiment by adjusting hyperparameters, adding more layers, or trying different architectures to improve accuracy.





