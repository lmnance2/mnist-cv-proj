import tensorflow
from tensorflow import keras
from keras import datasets
from keras import layers, models, utils



(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


y_train = utils.to_categorical(y_train, 10)
y_test  = utils.to_categorical(y_test, 10)


model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu", padding="same"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())

model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))

model.compile(
  optimizer="adam",
  loss="categorical_crossentropy",
  metrics=['accuracy']
)

history = model.fit(
  X_train, y_train, 
  epochs=25,
  batch_size=32
)

test_loss, test_acc = model.evaluate(X_test, y_test)