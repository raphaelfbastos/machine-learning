from model import Model
from layer import Dense
from activation import Tanh, Linear
from loss import MeanSquaredError
from pandas import read_csv

# Read in housing data.
df = read_csv(__file__.rsplit('\\', 1)[0] + '/Housing.csv')

# Separate and normalize inputs and outputs.
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']].to_numpy()
y = df[['price']].to_numpy()
X_mean = X.mean(0)
X_std = X.std(0)
y_mean = y.mean()
y_std = y.std()
normalized_X = (X - X_mean) / X_std
normalized_y = (y - y_mean) / y_std

# Define network architecture.
network = Model(
    Dense(5, 10, Tanh),
    Dense(10, 1, Linear)
)

# Train network.
for _ in range(10000):
    network.backward(network.forward(normalized_X), normalized_y, loss=MeanSquaredError, learning_rate=1e-3)

# Export model.
network.export(__file__.rsplit('\\', 1)[0])
