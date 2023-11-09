import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
import unittest
import traffic

def get_model_param(conv_num=1,conv_filter_size=64,conv_kernel_size=3,conv_activation="relu",pool_size=2,hidden_num=0,hidden_dense=128,hidden_activation="relu",hidden_dropout=0.5,optimizer="adam"):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential()
    
    # Convolutional layers
    for i in range(conv_num):
        model.add(tf.keras.layers.Conv2D(
                conv_filter_size, (conv_kernel_size,conv_kernel_size), activation=conv_activation, input_shape=(traffic.IMG_WIDTH, traffic.IMG_HEIGHT, 3)
            ))

        # Max-pooling layer, using 2x2 pool size
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size,pool_size)))

    # Flatten units
    model.add(tf.keras.layers.Flatten())

    # Add a hidden layers with dropout
    for i in range(hidden_num):
        model.add(tf.keras.layers.Dense(hidden_dense, activation=hidden_activation))
        model.add(tf.keras.layers.Dropout(hidden_dropout))

    # Add an output layer with NUM_CATEGORIES
    model.add(tf.keras.layers.Dense(traffic.NUM_CATEGORIES, activation="softmax"))

    model.compile(
        optimizer=optimizer,
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

        results={}
        for num_conv in [2]:
            for conv_filter_size in [256]:
                for conv_kernel_size in [5]:
                    for pool_size in [2]:
                        for hidden_num in [1]:
                            for hidden_dense in [128] if hidden_num>0 else [0]:
                                for hidden_dropout in [0.3] if hidden_num>0 else [0]:
                                    #print(f"Trying: {num_conv};{conv_filter_size};{conv_kernel_size};{pool_size};{hidden_num};{hidden_dense};{hidden_dropout}")
                                    # Get a compiled neural network
                                    model = get_model_param(conv_num=num_conv,conv_filter_size=conv_filter_size,conv_kernel_size=conv_kernel_size,pool_size=pool_size,hidden_num=hidden_num,hidden_dense=hidden_dense,hidden_dropout=hidden_dropout)

                                    # Fit model on training data
                                    model.fit(x_train, y_train, epochs=traffic.EPOCHS,verbose=0)

                                    # Evaluate neural network performance
                                    eval = model.evaluate(x_test,  y_test, verbose=0)
                                    #print(f"-> Got: {eval[0]};{eval[1]}")
                                    print(f"{traffic.NUM_CATEGORIES};{num_conv};{conv_filter_size};{conv_kernel_size};{pool_size};{hidden_num};{hidden_dense};{hidden_dropout};;;{eval[0]};{eval[1]}")
                                    results[(traffic.NUM_CATEGORIES,num_conv,conv_filter_size,conv_kernel_size,pool_size,hidden_num,hidden_dense,hidden_dropout)] = eval
        print(results)
if __name__ == "__main__":
    unittest.main()
