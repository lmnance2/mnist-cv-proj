from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

datagen = ImageDataGenerator(
  rotation_range=15,
  width_shift_range=0.1,
  height_shift_range=0.1,
  horizontal_flip = True
)

datagen.fit(X_train)

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(32,32,3)))

model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))

model.compile(
  optimizer="adam",
  loss="categorical_crossentropy",
  metrics=['accuracy']
)

model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=30)

test_loss, test_acc = model.evaluate(X_test, y_test)