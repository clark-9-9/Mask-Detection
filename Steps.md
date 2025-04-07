1. create environment `python -m venv env` , pthon version 3.12.6

2. `pip install tensorflow keras imutils numpy opencv-python matplotlib argparse scipy scikit-learn pillow streamlit onnx tf2onnx`

3. training models

    - `python train_mask_detector.py --dataset dataset`

4. image mask detection commands
    - `python detect_mask_image.py --image images/pic1.jpeg --model mask_detector.h5`
5. video mask detection commands
    - `python detect_mask_video.py --model mask_detector.h5`
