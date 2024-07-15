# Vehicle Cut-In Detection using YOLO and LSTM with Attention

This project aims to detect vehicle cut-in events using YOLOv5 for object detection and a multitask LSTM model with attention mechanisms for sequence prediction. The project is implemented using Python, PyTorch, and TensorFlow.

## Features

1. Google Drive Integration
2. YOLOv5 Object Detection
3. Velocity Calculation
4. Feature Extraction
5. Label Generation
6. Sequence Creation
7. Multitask LSTM Model with Attention
8. Hyperparameter Tuning
9. Model Training and Evaluation
10. Warning System
11. Attention Visualization
12. Model Saving

 Usage

1. Mount Google Drive:
   - Ensure your Google Drive is mounted to access datasets and save results.

2. Run Object Detection:
   - Execute the object detection section to perform inference using YOLOv5.

3. Feature Extraction and Label Generation:
   - Run the sections for calculating velocities, extracting features, and generating labels.

4. Train the Model:
   - Train the multitask LSTM model with the extracted features and generated labels.

5. Evaluate the Model:
   - Evaluate the trained model on the test dataset and print the results.

6. Save and Use the Model:
   - Save the trained model to Google Drive.
   - Use the warning system to predict cut-in events and send alerts.

 Dependencies

- Python 3.x
- PyTorch
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- scikeras
- tqdm
- PIL (Pillow)

Installation
Install the required packages using pip:

```sh
pip install torch torchvision torchaudio
pip install tensorflow
pip install opencv-python-headless
pip install numpy matplotlib
pip install scikit-learn scikeras[tensorflow]
pip install tqdm
pip install pillow

This project detects events of vehicle cut-in by using YOLOv5 for object detection and a multitask LSTM model with attention mechanisms for sequence prediction.

Key Steps

1. Setup: Mount Google Drive and define paths for images, results, and YOLO weights.
2. Loading of Pre-Trained YOLOv5 Model: Download and set up the pre-trained YOLOv5 model for object detection.
3. Object Detection: inference on images; save detected results
4. Feature Extraction: compute velocities of objects from the detections to obtain features.
5. Label Generation: CUT-in detection and Direction prediction labels generation.
6. Sequence Preparation: Sequences of features ready for LSTM input.
7. Model Training: Train a multi-task LSTM with attention mechanisms on sequences.
8. Evaluation: Test the model. Compute and output relevant performance metrics
9. Visualization: Plot attention weights on a sample sequence
10. Warning System: Design a system to send warnings based on the model output.

Reference:
Dataset: https://idd.insaan.iiit.ac.in/
Yolov5 for object detection : https://github.com/ultralytics/yolov5
