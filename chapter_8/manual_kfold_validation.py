from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy


# Set a fix random number
numpy.random.seed(42)

# Load Pima Indians dataset
dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split features (first 8 columns) and labels (9th column)
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Define 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for train, test in kfold.split(Y, Y):
    # Create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cv_scores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cv_scores), numpy.std(cv_scores)))
