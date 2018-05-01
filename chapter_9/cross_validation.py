from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy


# Create the model required for KerasClassifier
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Set the random seed
numpy.random.seed(42)

# Load the dataset
dataset = numpy.loadtxt('../datasets/pima-indians-diabetes.csv', delimiter=',')

# Split features and labels
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Create the model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

# Evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
