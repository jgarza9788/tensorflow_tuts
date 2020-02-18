# https://www.tensorflow.org/tutorials/structured_data/time_series

#import some libs
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


#get the data
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

#for mat into dataframe
df = pd.read_csv(csv_path)
df.head()

# # show data
# with pd.option_context('display.max_rows', 10, 'display.max_columns', None): 
#     print(df)
    

#re package the data
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size 

    # print('start_index',start_index)
    # print('end_index',end_index)

    for i in range(start_index, end_index):
        # print('i',i)
        indices = range(i-history_size, i)
        # print('indices',indices)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


# train using the first 300,000 rows (about 2100 days)
TRAIN_SPLIT = 300000

tf.random.set_seed(13)

# print('\n\n')
# print(list(df.keys()))
# print('\n\n')
columns = list(df.keys())

# create a dataframe using just the temp in degree C
uni_data = df['T (degC)']
# uni_data = df[['T (degC)','sh (g/kg)']]
# uni_data = df[columns]
uni_data.index = df['Date Time']
uni_data.head()

# print(uni_data)

#make a graph and show it
uni_data.plot(subplots=True)
# plt.show()

#normalize data
uni_data = uni_data.values
# #https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/utils/normalize
# uni_data = tf.keras.utils.normalize(uni_data,-1,2)[0]
# print(*uni_data[0],sep='\n')

uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

uni_data = (uni_data-uni_train_mean)/uni_train_std

# print(uni_data[0][20])

univariate_past_history = 20
univariate_future_target = 0

#split the data into train and validation
x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,univariate_past_history,univariate_future_target)

x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,univariate_past_history,univariate_future_target)

print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])


def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()

    # print(min(plot_data[0]) ,max(plot_data[0]))

    ymin = min(plot_data[0])
    if ymin > plot_data[1]:
        ymin = plot_data[1]
    if ymin > plot_data[2]:
        ymin = plot_data[2]
    ymin -= 0.005

    ymax = max(plot_data[0])
    if ymax < plot_data[1]:
        ymax = plot_data[1]
    if ymax < plot_data[2]:
        ymax = plot_data[2]
    ymax += 0.005

    plt.ylim(ymin,ymax)
    # plt.ylim(auto=True )
    # plt.ylim('auto')
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

#plot and show
# show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
# plt.show()

def baseline(history):
    return np.mean(history) 

# show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,'Baseline Prediction Example')
# plt.show()

# Recurrent neural network
# A Recurrent Neural Network (RNN) is a type of neural network well-suited to time series data. RNNs process a time series step-by-step, maintaining an internal state summarizing the information they've seen so far. For more details, read the RNN tutorial. In this tutorial, you will use a specialized RNN layer called Long Short Term Memory (LSTM)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# print(train_univariate.shape)

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


#building the model
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 10

## train the model
# simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
#                     steps_per_epoch=EVALUATION_INTERVAL,
#                     validation_data=val_univariate, validation_steps=50)

# for x, y in val_univariate.take(3):
#     plot = show_plot([x[0].numpy(), y[0].numpy(),
#                     simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
#     plot.show()


features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']

features = df[features_considered]
features.index = df['Date Time']
features.head()

features.plot(subplots=True)

# plt.show()

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

def multivariate_data(dataset, target, start_index, end_index, history_size,
                        target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

    if single_step:
        labels.append(target[i+target_size])
    else:
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


past_history = 720
future_target = 72
STEP = 6

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,TRAIN_SPLIT, past_history,future_target, STEP,single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],TRAIN_SPLIT, None, past_history,future_target, STEP,single_step=True)

print ('Single window of past history : {}'.format(x_train_single[0].shape))

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

for x, y in val_data_single.take(1):
    print(single_step_model.predict(x).shape)


single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)

input()