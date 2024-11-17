# YOLOv11 Object Detection: Training and Inference with KITTI Dataset

This repository provides two scripts for training a YOLOv11 (here referred to as YOLOv11 in the code) model on the KITTI dataset and then using the trained model to detect objects in a video file.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Setup](#dataset-setup)
- [Training the Model](#training-the-model)
- [Detecting Objects in a Video](#detecting-objects-in-a-video)
- [Command-Line Arguments Summary](#command-line-arguments-summary)
- [Results and Outputs](#results-and-outputs)
- [Troubleshooting](#troubleshooting)

## Overview
The repository contains two scripts:

1. **`train_model.py`**: Automates the process of preparing the KITTI dataset for YOLO format, training a YOLOv11 model, and evaluating its performance.
2. **`detect_object.py`**: Uses the trained YOLOv11 model to detect objects in a given video file, annotating the video frames with bounding boxes and labels.

## Prerequisites
Make sure you have installed:
- Python 3.7 or above
- PyTorch (with CUDA support if you have an NVIDIA GPU)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library: `pip install ultralytics`
- OpenCV for Python: `pip install opencv-python`
- scikit-learn: `pip install scikit-learn`
- `kagglehub` (for the dataset import in the script)

## Dataset Setup
This example uses the [KITTI dataset](https://www.kaggle.com/klemenko/kitti-dataset). The `train_model.py` script uses `kagglehub` to download and set up the dataset. If you prefer to manually download and place the dataset, update the dataset path (`KITTI_BASE_DIR`).

1. **Automatic Download (Using `kagglehub`)**:
    - Ensure you have a Kaggle API key and properly configured environment.
    - The script `train_model.py` automatically uses `kagglehub` to download the dataset.

2. **Manual Download** (Optional):
    - Download the KITTI dataset from Kaggle.
    - Extract the dataset and place the files in a directory. Update `KITTI_BASE_DIR` in `train_model.py` to point to this directory.

## Training the Model
Follow these steps to train the YOLOv11 model on the KITTI dataset:

1. **Configure Training Parameters in `train_model.py`**:
   - `MODEL_ARCH`: Path to the YOLO model configuration file (e.g., `yolov11n.yaml`).
   - `EPOCHS`: Number of training epochs (default: 10).
   - `BATCH_SIZE`: Batch size for training (default: 16).
   - `IMG_SIZE`: Input image size (width and height).
   - `PROJECT_NAME`: Directory name under which training results will be saved.
   - `EXPERIMENT_NAME`: Subdirectory name under the project directory to store this experiment's results.

2. **Run Training Script**:
   ```bash
   python train_model.py
   ```
   The script will:
   - Download and prepare the KITTI dataset.
   - Convert KITTI annotations to YOLO format.
   - Split the dataset into training and validation sets.
   - Generate a `data.yaml` file for YOLO training.
   - Train the YOLOv11 model.
   - Validate the model and print metrics like precision, recall, and F1 score.

3. **Check Training Outputs**:
   - The trained model weights (`best.pt`) and logs will be saved in `PROJECT_NAME/EXPERIMENT_NAME/weights`.
   - Validation metrics and logs will be displayed in the console. The `best.pt` weights represent the best model observed during training.

**Example Output**:
- The script prints progress and metrics. After training, you should see lines like:
  ```
  Training completed!
  ...
  Validation completed!
  Validation Metrics:
  ...
  ```

## Detecting Objects in a Video
Once the model is trained, use the `detect_object.py` script to detect objects in a video.

### Steps to Run:
1. **Ensure Model Weights**:
   - The `best.pt` (or any trained `.pt` file) is present in the directory where the script is run or specify its path using the `--weights` argument.

2. **Run Inference on a Video**:
   ```bash
   python detect_object.py --input path/to/input_video.mp4
   ```
   The script reads the video from `path/to/input_video.mp4`, performs object detection frame by frame, and outputs an annotated video named `input_video_annotated.mp4` in the same directory by default.

**Key Arguments**:
- `--input`: Path to the input video file (required).
- `--output`: Path to the output annotated video file. If not specified, the script appends `_annotated` to the input filename.
- `--weights`: Path to the trained YOLO model weights file. Default is `best.pt`.
- `--conf-threshold`: Confidence threshold for detections (default: `0.25`).
- `--iou-threshold`: IoU threshold for NMS (default: `0.45`).
- `--show-live`: If set, displays the video with annotations as it processes.

**Example Command**:
```bash
python detect_object.py --input input_video.mp4 --weights YOLOv11-KITTI/exp1/weights/best.pt --show-live
```
This uses the trained weights at `YOLOv11-KITTI/exp1/weights/best.pt`, performs detection on `input_video.mp4`, displays annotated frames in real-time, and saves the annotated video.

## Command-Line Arguments Summary

### `train_model.py`
The script does not require command-line arguments as it is configured within the code. Adjust the hyperparameters in the script directly.

### My best model weights

Download my best model weights from [here](https://drive.google.com/file/d/1UY_u3iMTglbxarzjccV-AxpHQQtOXqTh/view?usp=sharing)

### `detect_object.py`
```bash
python detect_object.py --input VIDEO_FILE.mp4 [--output OUTPUT_FILE.mp4] [--weights PATH_TO_BEST.pt] [--conf-threshold 0.25] [--iou-threshold 0.45] [--show-live]
```
**Arguments**:
- `--input`: **(Required)** Path to input video file.
- `--output`: Path to save the annotated video file. Defaults to appending `_annotated` before the file extension of `--input`.
- `--weights`: Path to the YOLO model weights file (`.pt`). Defaults to `best.pt`.
- `--conf-threshold`: Confidence threshold (default: `0.25`).
- `--iou-threshold`: IoU threshold for NMS (default: `0.45`).
- `--show-live`: Display processed frames in a window as the script runs. Press `q` to stop processing early.

## Results and Outputs
- **Training Outputs**:
  - Directory `PROJECT_NAME/EXPERIMENT_NAME/weights` contains:
    - `best.pt`: Model weights of the best-performing epoch.
    - `last.pt`: Model weights of the final epoch.
  - Validation metrics and logs printed in the console.

- **Video Inference Outputs**:
  - The annotated video file is saved at the specified `--output` location or with `_annotated` appended to the input file name.
  - The script prints status messages during processing and confirms the output file location upon completion.
  - Optionally, if `--show-live` is used, a window displays real-time annotation.

## Troubleshooting
- **No GPU Available**: If `device = 'cpu'` is printed, ensure your environment has GPU support or that CUDA is installed. The model will still run on CPU but may be slower.
- **File Not Found**: If the dataset or input video paths are incorrect, ensure paths are correct and files exist.
- **Inconsistent Results**: Adjust hyperparameters like `BATCH_SIZE`, `EPOCHS`, `conf_threshold`, and `iou_threshold` to match your dataset characteristics.
- **Missing Dependencies**: If modules like `cv2` or `ultralytics` are not found, install them using `pip install`.

**Note**: The script names or the YOLO version references may differ. Ensure consistency in the model's `.pt` file usage and the YOLO version configured in `MODEL_ARCH`.

By following this README, you can successfully train a YOLOv11 model using the KITTI dataset and then detect objects in any given video file using the trained model.