from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy


# Set a fix random number
numpy.random.seed(42)

# Load Pima Indians dataset
dataset = numpy.loadtxt('../datasets/pima-indians-diabetes.csv', delimiter=',')

# Split features (first 8 columns) and labels (9th column)
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Split the model in train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10)
