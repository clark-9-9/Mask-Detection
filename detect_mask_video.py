# USAGE
# python detect_mask_video.py

# ============================================
# STEP 1: Import Required Libraries
# ============================================
# Import the necessary packages for deep learning, image processing, and video capture
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils.video import VideoStream
import time
import cv2
import os

# ============================================
# STEP 2: Define Face Mask Detection Function
# ============================================
# This function processes a frame to detect faces and predict if they're wearing masks
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and create a blob
    # A blob is a pre-processed image that's properly formatted for the neural network
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))  # Resize to 300x300 and normalize pixel values

    # Pass the blob through the network and get face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()  # Returns an array with face detection results

    # Initialize lists for storing face data and predictions
    faces = []  # Will store the extracted face images
    locs = []   # Will store the face bounding box coordinates
    preds = []  # Will store the mask/no-mask predictions

    # Loop over all detected faces
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) of the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by comparing with minimum confidence threshold
        if confidence > args["confidence"]:
            # Calculate the (x, y)-coordinates of the face bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box falls within the frame dimensions
            # This prevents index errors when extracting the face region
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face region from the frame
            # Process it for input to the mask detector model
            face = frame[startY:endY, startX:endX]
            if face.size > 0:  # Make sure the face region isn't empty
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert colors: BGR to RGB
                face = cv2.resize(face, (224, 224))           # Resize to 224x224 for the mask model
                face = img_to_array(face)                     # Convert to numpy array
                face = preprocess_input(face)                 # Apply MobileNetV2 preprocessing

                # Add the processed face and its location to our lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # Only make predictions if at least one face was detected
    if len(faces) > 0:
        # Convert the faces list to a numpy array for batch prediction
        faces = np.array(faces, dtype="float32")
        try:
            # Try standard Keras model prediction first
            preds = maskNet.predict(faces, batch_size=32)
        except AttributeError:
            # If that fails, try direct model call (for TensorFlow SavedModel format)
            preds = maskNet(faces).numpy()
        except Exception as e:
            # Handle any other prediction errors
            print(f"Error during prediction: {e}")
            preds = []

    # Return the face locations and their corresponding mask predictions
    return (locs, preds)

# ============================================
# STEP 3: Parse Command Line Arguments
# ============================================
# Set up command-line options for the script
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# ============================================
# STEP 4: Load Face Detection Model
# ============================================
# Load the pre-trained face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])  # Model architecture file
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])  # Model weights file
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)  # Initialize the model using OpenCV's DNN module

# ============================================
# STEP 5: Load Face Mask Detector Model
# ============================================
# Load the trained mask detector model with support for multiple model formats
print("[INFO] loading face mask detector model...")
model_path = args["model"]

# Try different model loading approaches based on file format
if os.path.isdir(model_path):
    # Load as a TensorFlow SavedModel directory
    print(f"[INFO] Loading as SavedModel directory: {model_path}")
    import tensorflow as tf
    maskNet = tf.saved_model.load(model_path)
elif model_path.endswith(".h5"):
    # Load as an H5 model file
    print(f"[INFO] Loading as H5 model: {model_path}")
    maskNet = load_model(model_path)
elif model_path.endswith(".keras"):
    # Load as a Keras model file
    print(f"[INFO] Loading as Keras model: {model_path}")
    maskNet = load_model(model_path)
else:
    # Try to load as H5 file regardless of extension
    try:
        print(f"[INFO] Attempting to load as H5 model: {model_path}")
        maskNet = load_model(model_path)
    except:
        # Handle model loading errors with helpful messages
        print(f"[INFO] Converting model file to H5 format first...")
        print(f"[INFO] Original model path: {os.path.abspath(model_path)}")
        
        h5_path = "mask_detector.h5"
        if not os.path.exists(h5_path):
            print(f"[WARNING] Please convert your model to .h5 format and save as {h5_path}")
            print(f"[WARNING] Attempting to continue, but may fail...")
        
        try:
            # Last attempt to load the model
            maskNet = load_model(model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[INFO] Please check if you have a compatible model file.")
            print("[INFO] You can use a pre-trained model from: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/face_detector")
            exit(1)  # Exit with error code

# ============================================
# STEP 6: Initialize Video Stream
# ============================================
# Start video capture from the webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()  # Initialize video stream from default camera (index 0)
time.sleep(2.0)  # Pause to allow camera sensor to warm up

# ============================================
# STEP 7: Main Processing Loop
# ============================================
# Process frames continuously until user quits
while True:
    # Grab a frame from the video stream and resize it
    frame = vs.read()
    if frame is None:
        print("[ERROR] Could not read frame from camera")
        break
        
    frame = imutils.resize(frame, width=400)  # Resize for faster processing

    # Detect faces and predict if they're wearing masks
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Process each detected face
    for (box, pred) in zip(locs, preds):
        # Extract the face bounding box coordinates
        (startX, startY, endX, endY) = box
        
        # Handle different prediction output formats (for compatibility)
        if isinstance(pred, np.ndarray) and len(pred) == 2:
            # Standard format: array with two values [mask_prob, no_mask_prob]
            (mask, withoutMask) = pred
        elif hasattr(pred, "mask") and hasattr(pred, "withoutMask"):
            # Alternative format: object with mask and withoutMask attributes
            mask = pred.mask
            withoutMask = pred.withoutMask
        else:
            # Skip this face if prediction format is unknown
            print(f"[WARNING] Unexpected prediction format: {type(pred)}")
            continue

        # Determine class label and color based on prediction
        # Green for mask, red for no mask
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
        # Include the probability percentage in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Draw the label and bounding box on the frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display the processed frame in a window
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF  # Check for key press (1ms delay)

    # If 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# ============================================
# STEP 8: Cleanup
# ============================================
# Release resources when done
cv2.destroyAllWindows()  # Close all OpenCV windows
vs.stop()  # Stop video stream