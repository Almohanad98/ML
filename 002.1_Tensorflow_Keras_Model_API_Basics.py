import tensorflow as tf
import numpy as np

X = tf.random.uniform((100, 1))
y = (X + 5) * 0.3

X.shape, y.shape

model =  tf.keras.Sequential([
    tf.keras.layers.Input(name="InputLayer", shape=[1]),
    tf.keras.layers.Dense(100, name="Hidden_Layer_1"),
    tf.keras.layers.Dense(50, name="Hidden_Layer_2"),
    tf.keras.layers.Dense(10, name="Hidden_Layer_3"),
    tf.keras.layers.Dense(1, name="Output_Layer"),
    ])
model.summary()


tf.keras.utils.plot_model(model)

tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="TB", dpi=96)

model.compile(loss="MSE", optimizer="adam")

model.fit(X, y, epochs=10,)

model.predict(np.array([[9]]))

(np.array([9]) + 5) * 0.3

X_test = tf.random.uniform((10, 1))

model.predict(X_test)