import csv
import os
import sys
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

cwd = os.path.abspath(os.path.dirname(sys.argv[0]))
modelPath = 'MyModel_tf'
pathTest = cwd + "/../Participants_Data_HPP/Test.csv"

features = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT", "READY_TO_MOVE", "RESALE", "LONGITUDE", "LATITUDE", "TARGET(PRICE_IN_LACS)"]

# get test dataset
house_price_test = pd.read_csv(pathTest)[features]

house_price_test_features = house_price_test.copy()
# pop column
house_price_test_expected = house_price_test_features.pop('TARGET(PRICE_IN_LACS)')

# load model
new_model = tf.keras.models.load_model(modelPath)

# Check its architecture
# new_model.summary()

# Evaluate the restored model
loss = new_model.evaluate(house_price_test_features, house_price_test_expected, verbose=2)
print("------\n")
print(f"loss result: {loss}\n")
print("------")

#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

count = 0
try: 
    with open('trainResults.csv', 'r') as trainResults:
        count = sum(1 for _ in trainResults)
except:
    pass

with open('trainResults.csv', 'a+') as trainResults:
  trainResults.write(f"{count},{loss}" + "\n")

try: 
    x = []
    y = []
    with open('trainResults.csv', 'r') as trainResults:
        plots = csv.reader(trainResults, delimiter = ',')
        for row in plots:
            x.append(row[0])
            y.append(float(row[1]))
        plt.bar(x, y, color = 'g', label = "loss")
        plt.xlabel('builds')
        plt.ylabel('losses')
        plt.title('loss for build')
        plt.legend()
        plt.ylim(ymin=min(y), ymax=max(y))
        plt.savefig('metrics.png')
         
except:
    pass