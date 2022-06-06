import os
import sys
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#train params
numberOfEpoch = sys.argv[1]

cwd = os.path.abspath(os.path.dirname(sys.argv[0]))

pathTrain = cwd + "./Participants_Data_HPP/Train.csv"
pathTest = cwd + "./Participants_Data_HPP/Test.csv"

features = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT", "READY_TO_MOVE", "RESALE", "LONGITUDE", "LATITUDE", "TARGET(PRICE_IN_LACS)"]

# get dataset
house_price_train = pd.read_csv(pathTrain)[features]

# get test dataset
house_price_test = pd.read_csv(pathTest)[features]


house_price_features = house_price_train.copy()
# pop column
house_price_labels = house_price_features.pop('TARGET(PRICE_IN_LACS)')

# process data
normalize = layers.Normalization()
normalize.adapt(house_price_features)

feature_test_sample = house_price_test.sample(10)
labels_test_sample = feature_test_sample.pop('TARGET(PRICE_IN_LACS)')

house_price_test_features = house_price_test.copy()
# pop column
house_price_test_expected = house_price_test_features.pop('TARGET(PRICE_IN_LACS)')

# to np.array
# house_price_test =  np.array(house_price_test)
# house_price_test_expected = np.array(house_price_test_expected)

house_price_features = np.array(house_price_features)

# checkoints
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
# model keras.Sequential
# one output tensor

modelPath = 'saved_model/MyModel_tf'
try: 
  linear_model = tf.keras.models.load_model(modelPath)
  print("open existing model")
except Exception as ex:
  print(ex)
  linear_model = tf.keras.Sequential([
    normalize,
    layers.Dense(1)
  ])
  linear_model.compile(loss = tf.losses.MeanSquaredError(),
                        optimizer = tf.optimizers.Adam(1))
  print("creating new model")

# train model
history = linear_model.fit(
  house_price_features, 
  house_price_labels, 
  epochs=int(numberOfEpoch), 
  validation_split=0.33,
  verbose=1)
#callbacks=[cp_callback])

# save model
linear_model.save(modelPath, save_format='tf')

test_results = {}
test_results['linear_model'] = linear_model.evaluate(
    house_price_test_features, house_price_test_expected, verbose=0)

def flatten(t):
    return [item for sublist in t for item in sublist]

pred = np.array(linear_model.predict(feature_test_sample))
flatten_pred = flatten(pred)

# print("predictions: " + str(flatten_pred))
# print("expected: " + str(np.array(labels_test_sample)))

with open(cwd + "/../result.txt", "w+") as resultFile:
  resultFile.write("predictions: " + str(flatten_pred) + '\n')
  resultFile.write("expected: " + str(labels_test_sample.to_numpy()))
