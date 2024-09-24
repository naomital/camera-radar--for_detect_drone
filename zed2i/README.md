# Drone Detection System Using Optical Flow and Background Subtraction

## Overview
This Python code is designed to detect drones in a video frame using a combination of background subtraction and optical flow techniques. The system first isolates moving objects by subtracting the background, then tracks the motion of these objects to differentiate drones from other objects based on movement.

## Features
1. **Bounding Box Intersection Over Union (IoU)**: Calculates the overlap between bounding boxes to determine how much they intersect.
2. **Bounding Box Merging**: Merges multiple overlapping bounding boxes based on a threshold, ensuring more accurate detection.
3. **Optical Flow**: Tracks motion between video frames using the Lucas-Kanade method, enabling the system to detect drones based on movement patterns.
4. **Background Subtraction**: Uses background subtraction (MOG2 algorithm) to isolate moving objects in the video, filtering out static backgrounds such as the sky.
5. **Noise Reduction**: Morphological operations are applied to the background mask to reduce noise in the detected moving objects.
6. **Real-Time Visualization**: Displays the detected and tracked drone as a bounding box around the moving object in real-time.

## Main Components

### 1. IoU Between Bounding Boxes
```python
def iou_between_boxes(box1, box2):
```
- Calculates the Intersection over Union (IoU) between two bounding boxes.
- Helps determine how much two bounding boxes overlap to facilitate merging.

### 2. Merging Bounding Boxes
```python
def merge_bounding_boxes(bboxes, iou_threshold=0.05):
```
- Merges overlapping bounding boxes based on the IoU score. Bounding boxes with IoU above the specified threshold are combined into one.

### 3. Drone Detection
```python
def detect_drone_in_the_sky(frame, prev_gray, lk_params, fgbg, threshold_contour_rea=1, motion_threshold=0.5):
```
- Detects drones in the video frame by:
  - Applying background subtraction to isolate moving objects.
  - Using optical flow to track and validate the motion of detected objects.
  - Filtering objects based on their motion magnitude to identify drones.
  - Displays a bounding box around the detected drone and labels it as "moving" if its motion exceeds the defined threshold.

### 4. Main Function
```python
def main(video_path: str):
```
- The entry point for running the drone detection system.
- Reads frames from the video, processes each frame to detect drones, and displays the result in a real-time window.
- Press 'q' to exit the video window.

## How to Use

1. **Prerequisites**:
   - Install required libraries:
     ```bash
     pip install opencv-python numpy
     ```
   - Ensure you have a video file of a drone flying in the sky.

2. **Run the Code**:
   - Modify the `video_path` in the `main` function to point to your video file.
   - Run the script:
     ```bash
     python detect_drone_in_the_sky.py
     ```

3. **Output**:
   - The video will play in a window with the drone's bounding box highlighted.
   - Drones are labeled "moving" if their motion is detected between frames.

## Parameters

- **iou_threshold**: Threshold for merging bounding boxes based on overlap.
- **threshold_contour_rea**: Minimum contour area for detecting objects to avoid false positives from small, noisy areas.
- **motion_threshold**: Minimum motion magnitude required to classify a detected object as moving.

## Notes
- The system assumes a relatively clear sky, where the background subtraction can cleanly isolate the drone.
- The motion threshold and contour area can be tuned to improve detection accuracy based on specific video conditions.

