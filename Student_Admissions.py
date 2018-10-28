#!/usr/bin/env python
# coding: utf-8

# # Predicting Student Admissions
# 
# In this notebook, we predict student admissions to graduate school at UCLA based on three pieces of data:
# - GRE Scores (Test)
# - GPA Scores (Grades)
# - Class rank (1-4)
# 
# The dataset originally came from here: http://www.ats.ucla.edu/
# 
# _Note: Thanks Adam Uccello, for helping us debug!_

# ## 1. Load and visualize the data
# 
# To load the data, we will use a very useful data package called Pandas. You can read on Pandas documentation here:

# In[287]:


import pandas as pd
data = pd.read_csv('student_data.csv')
data


# Let's plot the data and see how it looks.

# In[288]:


import matplotlib.pyplot as plt
import numpy as np
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
plot_points(data)
plt.show()


# The data, based on only GRE and GPA scores, doesn't seem very separable. Maybe if we make a plot for each of the ranks, the boundaries will be more clear.

# In[289]:


data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()


# These plots look a bit more linearly separable, although not completely. But it seems that using a multi-layer perceptron with the rank, gre, and gpa as inputs, may give us a decent solution.

# # 2. Process the data
# We'll do the following steps to clean up the data for training:
# - One-hot encode the rank
# - Normalize the gre and the gpa scores, so they'll be in the interval (0,1)
# - Split the data into the input X, and the labels y.

# In[301]:


import keras
from keras.utils import np_utils

# remove NaNs
data = data.fillna(0)

# One-hot encoding the rank
processed_data = pd.get_dummies(data, columns=['rank'])

# Normalizing the gre and the gpa scores to be in the interval (0,1)
processed_data["gre"] = processed_data["gre"]/800
processed_data["gpa"] = processed_data["gpa"]/4

# Splitting the data input into X, and the labels y 
X = np.array(processed_data)[:,1:]
X = X.astype('float32')
y = keras.utils.to_categorical(data["admit"],2)


# In[302]:


# Checking that the input and output look correct
print("Shape of X:", X.shape)
print("\nShape of y:", y.shape)
print("\nFirst 10 rows of X")
print(X[:10])
print("\nFirst 10 rows of y")
print(y[:10])


# In[ ]:





# # 3. Split the data into training and testing sets

# In[292]:


# break training set into training and validation sets
(X_train, X_test) = X[50:], X[:50]
(y_train, y_test) = y[50:], y[:50]

# print shape of training set
print('x_train shape:', X_train.shape)

# print number of training, validation, and test images
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# # 4. Define the model architecture

# In[309]:


# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
# Note that filling out the empty rank as "0", gave us an extra column, for "Rank 0" students.
# Thus, our input dimension is 7 instead of 6.
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(7,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# # 5. Train the model

# In[310]:


# Training the model
model.fit(X_train, y_train, epochs=200, batch_size=100, verbose=0)


# # 6. Score the model

# In[308]:


# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train)
print("\n Training Accuracy:", score[1])
score = model.evaluate(X_test, y_test)
print("\n Testing Accuracy:", score[1])


# # 7. Play with parameters!
# You can see that we made several decisions in our training. For instance, the number of layers, the sizes of the layers, the number of epochs, etc.
# 
# It's your turn to play with parameters! Can you improve the accuracy? The following are other suggestions for these parameters. We'll learn the definitions later in the class:
# - Activation function: relu and sigmoid
# - Loss function: categorical_crossentropy, mean_squared_error
# - Optimizer: rmsprop, adam, ada

# In[ ]:





# In[ ]:




