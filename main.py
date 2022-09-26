import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
dataset = pd.read_csv('hiring.csv')
#print(dataset.head())

dataset['experience'].fillna(0, inplace=True)
#print(dataset.head())
dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(), inplace=True)
#print(dataset.head())

#Converting words to integer values
X = dataset.iloc[:, :3]
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]
#print(dataset.head())

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

import sklearn.linear_model
regressor = sklearn.linear_model.LinearRegression()

#Fitting model with trainig data
regressor.fit(X.values, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))

