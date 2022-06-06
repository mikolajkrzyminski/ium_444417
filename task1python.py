import os
import sys
import pandas as pd


# paths
filePathTest = "./Participants_Data_HPP/Train.csv"
filePathTrain = "./Participants_Data_HPP/Test.csv"

dataTest = pd.read_csv(filePathTest)
dataTrain = pd.read_csv(filePathTrain)

number_lines = len(dataTest.index)
row_size = number_lines // 2

# start looping through data writing it to a new file for each set
# no of csv files with row size
k = 2
size = row_size

# split test data to test and dev
for i in range(k):
    df = dataTest[size * i:size * (i + 1)]
    name = ""
    if i == 0:
        name = "Dev"
    else:
        name = "Test"
    df.to_csv(cwd + './Participants_Data_HPP/' + name + '.csv', index=False)

#df_1 = pd.read_csv("../Participants_Data_HPP/Dev.csv")

#df_2 = pd.read_csv("../Participants_Data_HPP/Test.csv")

#df_2 = pd.read_csv("../Participants_Data_HPP/Train.csv")

dataPath = './Participants_Data_HPP/Train.csv'

#data informations
data = pd.read_csv(dataPath)

description = data.describe(include="all")

corr = data.corr()

#select the most significant
data = data[['TARGET(PRICE_IN_LACS)', 'SQUARE_FT', 'BHK_NO.', 'RESALE']]
#normalize price column and flat area using min max technique
columnName1 = 'TARGET(PRICE_IN_LACS)'
columnName2 = 'SQUARE_FT'

column1Min = data[columnName1].min()
column1Max = data[columnName1].max()
column2Min = data[columnName2].min()
column2Max = data[columnName2].max()

data[columnName1] = (data[columnName1] - column1Min) / (column1Max - column1Min)
data[columnName2] = (data[columnName2] - column2Min) / (column2Max - column2Min)

print(description)

print(corr)

print(data.describe(include="all"))

print(data.head())
