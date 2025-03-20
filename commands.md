image mask detection commands

-   python detect_mask_image.py --image images/pic1.jpeg
-   python detect_mask_image.py --image images/pic1.jpeg --model mask_detector.h5

---

video mask detection commands

-   python detect_mask_video.py --model mask_detector.h5

---

training commands

-   python train_mask_detector.py --dataset dataset

---

after creating and activating the virtual environment

-   python.exe -m pip install --upgrade pip

---

pip install tensorflow keras imutils numpy opencv-python matplotlib argparse scipy scikit-learn pillow streamlit onnx tf2onnx

pip install tensorflow>=2.5.0 keras==2.4.3 imutils==0.5.4 numpy>=1.23.5 opencv-python>=4.2.0.32 matplotlib==3.4.1 argparse==1.4.0 scipy==1.6.2 scikit-learn==0.24.1 pillow>=8.3.2 streamlit==0.79.0 onnx==1.10.1 tf2onnx==1.9.3
