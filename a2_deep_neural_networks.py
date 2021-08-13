from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations

#model with multiple layers for deep neural networks
moel = keras.Sequential([
    #the hidden ReLu layer
    layers.Dense(units=4,activation="relu",input_shape=[3]), #relu - rectified linear unit function
    layers.Dense(units=3,activations="relu"),
    #the linear output layer
    layers.Dense(unites=1)

])