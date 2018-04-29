from keras.models import Sequential
from keras.layers import Dense
import numpy


# Set a fix random number
numpy.random.seed(42)

# Load Pima Indians dataset
dataset = numpy.loadtxt('../datasets/pima-indians-diabetes.csv', delimiter=',')

# Split features (first 8 columns) and labels (9th column)
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
