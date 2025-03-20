# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

def mask_image():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.h5",  # Changed default to .h5 extension
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model_path = args["model"]

    # Try different model loading approaches based on file extension
    if os.path.isdir(model_path):
        print(f"[INFO] Loading as SavedModel directory: {model_path}")
        import tensorflow as tf
        model = tf.saved_model.load(model_path)
    elif model_path.endswith(".h5"):
        print(f"[INFO] Loading as H5 model: {model_path}")
        model = load_model(model_path)
    elif model_path.endswith(".keras"):
        print(f"[INFO] Loading as Keras model: {model_path}")
        model = load_model(model_path)
    else:
        # Try to load as H5 file regardless of extension
        try:
            print(f"[INFO] Attempting to load as H5 model: {model_path}")
            model = load_model(model_path)
        except:
            print(f"[INFO] Converting model file to H5 format first...")
            print(f"[INFO] Original model path: {os.path.abspath(model_path)}")
            
            # If you have the original model, you can convert it
            # This is a placeholder - you may need to create this file manually
            h5_path = "mask_detector.h5"
            if not os.path.exists(h5_path):
                print(f"[ERROR] Please use the H5 format model with --model mask_detector.h5")
                exit(1)
            
            try:
                # Try to load the h5 model instead
                model = load_model(h5_path)
                print(f"[INFO] Successfully loaded model from: {h5_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                print("[INFO] Please check if you have a compatible model file.")
                exit(1)

    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread(args["image"])
    if image is None:
        print(f"[ERROR] Could not read image: {args['image']}")
        exit(1)
    
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            try:
                # Try to use predict() method (for regular Keras models)
                preds = model.predict(face)
                (mask, withoutMask) = preds[0]
            except AttributeError:
                # If not a regular model (e.g., TF SavedModel), use direct call
                preds = model(face).numpy()
                (mask, withoutMask) = preds[0]
            except Exception as e:
                print(f"[ERROR] Prediction error: {e}")
                continue

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    mask_image()