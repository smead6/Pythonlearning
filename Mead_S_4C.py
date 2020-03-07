# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:19:54 2020

@author: Samuel
"""
import pandas as pd
URL = 'https://raw.githubusercontent.com/smead6/Pythonlearning/master/breast-cancer.csv'


dataset = pd.read_csv(URL,sep=',')

#Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Importing the dataset

#dataset = pd.read_csv('breast-cancer.csv')


"""It seemed easier with the amount of variables which had to be encoded to do
it dynamically accross the spreadsheet - faffing with labelencoders for 
nearly every Var didn't seem to be part of the exercise.......""" 
# Encoding categorical data
le= LabelEncoder()
for column_name in dataset.columns:
    if dataset[column_name].dtype==object:
        dataset[column_name]=le.fit_transform(dataset[column_name])
    else:
        pass
#assigning X and y
X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9].values

"""Splitting the dataset into the Training set and Test set - the same method 
was used as the udemy examples as it seemed to work, but then test_size and
random_state were ammended"""


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# Feature Scaling - as per Udemy example. If it ain't broke, don't fix it.....
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""
#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[100],
              'nb_epoch': [250],
              'optimizer':['adam']}
grid_search = GridSearchCV(estimator= classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 2)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

"""


# Importing the Keras libraries and packages
import keras # I left this here for the build to make runtime quicker when debugging
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
classifier.add(Dropout(p=0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 25, epochs = 250)



# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""
new_prediction = classifier.predict(sc.transform(np.array([[2,5,1,1,3,1,2,1,1]])))
new_prediction = (new_prediction > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('The result of the prediction is', new_prediction)
"""

import keras # I left this here for the build to make runtime quicker when debugging the evaluation part
from keras.models import Sequential
from keras.layers import Dense



"""
#Evaluating the CNN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 18, epochs = 150)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 18, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()
"""
"""
#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[100],
              'nb_epoch': [250],
              'optimizer':['adam']}
grid_search = GridSearchCV(estimator= classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 2)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_"""

"""Parameter optimizations: a stochastic method was used to work out which ones were best to start with.
This had the batch size as 5,10,25 and 32, nb_epoch at 100,250,325,500 and 750. Using adam and rmsprop.
Once the best batch size and nb_epoch came out as 5 and 100 the batch_size on iteration 2 were 3,5,8,10 and nb_epoch were 25,100,250.
On iteration 3 the parameters used was a batch size of 3, and nb_epoch  of 20 and 25. This was to attempt to prove the robustness of
each optomiser; as it was likely the SGD may have found local minima. 
Finally, the best optimizer was picked at a batch size of 25 and 250 epochs. """