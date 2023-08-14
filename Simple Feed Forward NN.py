import tensorflow as tf
import csv

from sklearn.model_selection import train_test_split

# read in data set
with open("") as f:
    reader = csv.reader(f)
    next(reader)

# format data set
    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0
        })

# divide into training and testing set
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size = 0.4
)

# neural network structure, sequential means one layer after the other
model = tf.keras.models.Sequential()

# hidden/dense layers, 8 units, ReLU activation function
model.add(tf.keras.layers.Dense(8, input_shape = (4,), activation = "relu"))

# add output layer, 1 unit, sigmoid activation
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# training and optimizing model
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuarcy"]
)
model.fit(X_training, y_training, epochs = 20)

# evaluation
model.evaluate(X_testing, y_testing, verbose = 2)


