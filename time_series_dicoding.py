# -*- coding: utf-8 -*-
"""Time-Series Dicoding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pelmz2Ntt9x0cj5X-Yxzy1dQibbDGImh
"""

import pandas as pd
data = pd.read_csv('/content/london_weather.csv',sep=',')

data.head()

data.shape

data.reset_index(drop=True, inplace=True)
data['step'] = data.index

data

data = data[['date', 'max_temp', 'step']]

data

data = data.dropna()

data.shape

"""##PLOT Series"""

import matplotlib.pyplot as plt

def plot_series(x, y, format="-", start=0, end=None, title=None, xlabel=None, ylabel=None, legend=None ):

    plt.figure(figsize=(14, 6))
    if type(y) is tuple:
      for y_curr in y:
        plt.plot(x[start:end], y_curr[start:end], format)
    else:
      plt.plot(x[start:end], y[start:end], format)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
      plt.legend(legend)
    plt.title(title)
    plt.grid(True)
    plt.show()

max_temp_time_step = list(data['step'])
max_temp_list = list(data['max_temp'])

import numpy as np
max_temp_time = np.array(max_temp_time_step)
max_temp_series = np.array(max_temp_list)

max_temp_series

plot_series(max_temp_time, max_temp_series, xlabel='day', ylabel='....')

"""##SPLIT DATASET"""

split_time = 12268 # 20%
time_train = max_temp_time[:split_time]
x_train = max_temp_series[:split_time]
time_valid = max_temp_time[split_time:]
x_valid = max_temp_series[split_time:]

time_train.shape

import tensorflow as tf

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

train_set

"""##MODELING"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                      strides=1,
                      activation="relu",
                      padding='causal',
                      input_shape=[window_size, 1]),
    # tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.summary()

init_weights = model.get_weights()
model.set_weights(init_weights)

threshold = (data['max_temp'].max() - data['max_temp'].min()) * 10/100
print(threshold)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mae')<4.41):
            print("\n MAE is less than 10% of data scale")
            self.model.stop_training = True


callbacks = myCallback()

learning_rate = 7e-6

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

history = model.fit(train_set,epochs=20, callbacks=[callbacks])

mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss))

plot_series(
    x=epochs,
    y=(mae, loss),
    title='MAE and Loss',
    xlabel='MAE',
    ylabel='Loss',
    legend=['MAE', 'Loss']
    )

zoom_split = int(epochs[-1] * 0.2)
epochs_zoom = epochs[zoom_split:]
mae_zoom = mae[zoom_split:]
loss_zoom = loss[zoom_split:]

plot_series(
    x=epochs_zoom,
    y=(mae_zoom, loss_zoom),
    title='MAE and Loss',
    xlabel='MAE',
    ylabel='Loss',
    legend=['MAE', 'Loss']
    )

def model_forecast(model, series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)
    return forecast

forecast_series = max_temp_series[split_time-window_size:-1]

forecast = model_forecast(model, forecast_series, window_size, batch_size)

results = forecast.squeeze()

plot_series(time_valid, (x_valid, results))

