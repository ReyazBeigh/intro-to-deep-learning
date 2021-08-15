from numpy.lib.function_base import disp
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from IPython.display import display
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import History
import matplotlib.pyplot as plt

csv_input = "./inputs/winequality-red.csv"
red_wine_data = pd.read_csv(csv_input)

# create training and validation splits
df_train = red_wine_data.sample(frac=0.7, random_state=0)
df_test = red_wine_data.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)

min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_test = (df_test - min_) / (max_ - min_)

# Split features and target

X_train = df_train.drop(['quality'], axis=1)
X_test = df_test.drop(['quality'], axis=1)
y_train = df_train['quality']
y_test = df_test['quality']

model = keras.Sequential([
    layers.Dense(1024, activation="relu", input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

model.compile(optimizer="adam", loss="mae")

history = model.fit(X_train, y_train, epochs=100, batch_size=256,
                    validation_data=(X_test, y_test), verbose=0)

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot();