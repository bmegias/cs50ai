import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
import unittest
import traffic

def get_model_param():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential()
    
    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    model.add(tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(traffic.IMG_WIDTH, traffic.IMG_HEIGHT, 3)
        ))

    # Max-pooling layer, using 2x2 pool size
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten units
    model.add(tf.keras.layers.Flatten())

    # Add a hidden layer with dropout
    #model.add(model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    # Add an output layer with NUM_CATEGORIES
    model.add(tf.keras.layers.Dense(traffic.NUM_CATEGORIES, activation="softmax"))


    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

class TestTraffic(unittest.TestCase):
    def test_models(self):
        images, labels = traffic.load_data("gtsrb")
    
        # Split data into training and testing sets
        labels = tf.keras.utils.to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=traffic.TEST_SIZE
        )

        # Get a compiled neural network
        model = get_model_param()

        # Fit model on training data
        model.fit(x_train, y_train, epochs=traffic.EPOCHS)

        # Evaluate neural network performance
        eval = model.evaluate(x_test,  y_test, verbose=2)
        print(eval)
        
if __name__ == "__main__":
    unittest.main()
