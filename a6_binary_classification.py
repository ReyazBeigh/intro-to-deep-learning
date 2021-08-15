from numpy.lib.function_base import disp
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from IPython.display import display

csv_input = "./inputs/ion.csv"

ion_data = pd.read_csv(csv_input,index_col=0)

#display(ion_data.head())

df = ion_data.copy()
df['Class'] = df['Class'].map({'good': 0, 'bad': 1})

df_train = df.sample(frac=0.7, random_state=1)
df_valid = df.drop(df_train.index)

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
#display(min_)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

df_train.dropna(axis=1, inplace=True)  # drop the empty feature in column 2
df_valid.dropna(axis=1, inplace=True)

X_train = df_train.drop(['Class'], axis=1)
X_valid = df_valid.drop(['Class'], axis=1)

y_train = df_train['Class']
y_valid = df_valid['Class']


model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['binary_accuracy'])

early_stopping = EarlyStopping(
    min_delta=0.001, patience=10, restore_best_weights=True)


history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=512, callbacks=[early_stopping],
                    verbose=False,  # hide the output because we have so many epochs
                    )

history_df = pd.DataFrame(history.history)

history_df.loc[5:,['loss','val_loss']].plot()
history_df.loc[5:,['binary_accuracy','val_binary_accuracy']].plot()

print("Best Validation Loss: {:0.4f}\nBest Validation Accuracy: {:0.4f}".format(history_df['val_loss'].min(), history_df['val_binary_accuracy'].max()))