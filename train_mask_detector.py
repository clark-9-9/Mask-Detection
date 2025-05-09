# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector.h5", help="path to output face mask detector model")
args = vars(ap.parse_args())

# initialize training parameters
INIT_LR = 1e-4  # Initial learning rate for optimizer (helps control step size during weight updates)
EPOCHS = 20     # Number of passes through the full training dataset
BS = 32         # Batch size: number of samples processed before the model updates

# ============================================
# STEP 1: Load all image file paths and labels
# ============================================
# Walk through the dataset directory and collect all image paths with valid extensions
print("[INFO] loading images...")
imagePaths = []  # List to store complete image file paths
data = []        # List to hold image data after processing
labels = []      # List to hold corresponding class labels

for root, dirs, files in os.walk(args["dataset"]):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            imagePaths.append(os.path.join(root, file))

# ============================================
# STEP 2: Load and process each image
# ============================================
# Read each image, resize it, convert to array, and preprocess for MobileNetV2
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]  # Label is folder name: with_mask / without_mask
    image = load_img(imagePath, target_size=(224, 224))  # Resize image to fit MobileNetV2 input
    image = img_to_array(image)                          # Convert image to array
    image = preprocess_input(image)                      # Normalize using MobileNetV2 preprocessing
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")   # Final image array (float32 for deep learning models)
labels = np.array(labels)                # Convert label list to array

# ============================================
# STEP 3: One-hot encode the labels
# ============================================
# Convert categorical labels into one-hot encoded vectors
lb = LabelBinarizer()
labels = lb.fit_transform(labels)        # Transform to binary labels (0 or 1)
labels = to_categorical(labels)          # One-hot encode for softmax output layer

# ============================================
# STEP 4: Split data into training and testing
# ============================================
# Randomly split data into 80% training and 20% testing (with class stratification)
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# ============================================
# STEP 5: Image data augmentation
# ============================================
# Apply random transformations to increase training data diversity

# Image data augmentation creates new training images by randomly 
# transforming existing ones (like rotating, zooming, flipping, shifting). 
# It helps prevent overfitting and makes the model more robust to variations.

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ============================================
# STEP 6: Load MobileNetV2 base model
# ============================================
# MobileNetV2 is a lightweight, pre-trained deep learning model designed for (image recognition). 
# It’s fast and efficient, making it great for mobile or real-time applications.

# In this code, it’s used without the top layers to act as a feature extractor—it 
# learns patterns (edges, textures, etc.) from images, which are then passed to a custom classifier head.

# Load pre-trained MobileNetV2 without the top layers (for feature extraction)

baseModel = MobileNetV2(
    weights="imagenet",           # Use weights trained on ImageNet
    include_top=False,            # Remove original classification head
    input_tensor=Input(shape=(224, 224, 3))  # Define new input shape
)

# ============================================
# STEP 7: Build the classification head
# ============================================
# Construct new layers to perform classification based on extracted features

# The classification head is needed because it transforms the features extracted 
# by the base model (like MobileNetV2) into a final decision or output that the model can use to make predictions(output).

# classification head in deep learning is the part of the model that 1- makes predictions(output, result). 
# 2- It processes features extracted by the base model (like MobileNetV2) and outputs the final classification

# In deep learning, (features) refer to the patterns, characteristics, or representations 
# learned from the input data (like images, text, etc.).

# In deep learning, a (prediction) refers to the (output or result) produced by a model after it processes input data.

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)  # Pool down feature maps
headModel = Flatten(name="flatten")(headModel)             # Flatten into 1D vector
headModel = Dense(128, activation="relu")(headModel)       # Fully connected layer
headModel = Dropout(0.5)(headModel)                        # Drop 50% of neurons randomly
headModel = Dense(2, activation="softmax")(headModel)      # Output layer with 2 class probabilities

# ============================================
# STEP 8: Combine base and head into full model
# ============================================
# Merge base and new head into a single model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model layers to preserve learned features during initial training
for layer in baseModel.layers:
    layer.trainable = False

# ============================================
# The base model extracts features() from the input.

# The classification head uses those features to make the final prediction (e.g., identifying the class or category).

# ============================================
# STEP 9: Compile the model
# ============================================
# Prepare model for training by specifying loss, optimizer, and metric
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)  # Adaptive learning optimizer
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# ============================================
# STEP 10: Train the model
# ============================================
# Train the model on training data and evaluate on validation data
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),              # Apply real-time data augmentation
    steps_per_epoch=len(trainX) // BS,                    # Number of batches per epoch
    validation_data=(testX, testY),                       # Validation data
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# ============================================
# STEP 11: Evaluate model performance
# ============================================
# Use the test set to evaluate how well the model performs
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)            # Predict class probabilities
predIdxs = np.argmax(predIdxs, axis=1)                    # Convert to class labels (0 or 1)

# Print a classification report: precision, recall, f1-score, etc.
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# ============================================
# STEP 12: Save model to disk
# ============================================
# Save the trained model for future use or deployment
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")                   # Save in HDF5 format
model.save(args["model"].replace(".h5", ".keras"))            # Also save in Keras format

# ============================================
# STEP 13: Plot training loss and accuracy
# ============================================
# Visualize the training process to detect overfitting or underfitting
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])  # Save the plot image to disk
