# importing the required packages

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical


# locating the folders
DIRECTORY = r"D:\FaceRecognition\dataset"
CATEGORIES = ["mask", "without mask"]

print("Loading Images...")

# Appending the image arrays to this empty list
image_arrays = []

# Appending the corresponding images which are with masks or without masks
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)

    # will list down all the images inside the specific category
    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        # loads all the images to a fixed target size
        image = load_img(img_path, target_size=(300, 300))

        # converting image to array
        image = img_to_array(image)

        # mobilenet function
        image = preprocess_input(image)

        # appending the image and category to the above empty lists
        image_arrays.append(image)
        labels.append(category)

        # converts the categorical values into new binary variable also known as one-hot encoding
        # https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(image_arrays, dtype="float32")
labels = np.array(labels)

# train-test-split method used.
# 0.2 means 20% for testing and remaining 80% for training the model
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=30)

# construct the training image generator for data augmentation
# ImageDataGenerator is used to make multiple images of a single image by changing its properties

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.15,)

# load the MobileNetV2 network, ensuring the head FC layer sets are
# imagenet is pretrained model
# 3 stands for RGB images
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(300, 300, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
    layer.trainable = False

INIT_LR = 1e-4
EPOCHS = 18
BS = 30

# compile our model
print("Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("Training head...")
H = model.fit(
    gen.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# serialize the model to disk
print("Saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
