import os
import shutil
import yaml
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# -----------------------------
# 1. Setup and Configuration
# -----------------------------

KITTI_BASE_DIR = '/kaggle/input/kitti-dataset'
"""str: The base directory where the KITTI dataset is located."""

IMAGE_DIR = Path(KITTI_BASE_DIR) / 'data_object_image_2' / 'training' / 'image_2'
"""Path: Directory containing KITTI training images."""

LABEL_DIR = Path(KITTI_BASE_DIR) / 'data_object_label_2' / 'training' / 'label_2'
"""Path: Directory containing KITTI training labels."""

TRAIN_DIR = Path('train')
"""Path: Directory where training images and labels will be stored in YOLO format."""

VALID_DIR = Path('valid')
"""Path: Directory where validation images and labels will be stored in YOLO format."""

LABELS_DIR = Path('labels_with_dont_care')
"""Path: Directory where YOLO-formatted labels will be stored."""

CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
    'Cyclist', 'Tram', 'Misc', 'DontCare'
]
"""
list of str: List of class names included in the KITTI dataset. 
'CLASSES' should reflect all possible object categories for detection.
"""

MODEL_ARCH = 'yolo11x.yaml'
"""str: The YOLO11x model configuration file to use."""

EPOCHS = 100
"""int: Number of epochs for training. Adjust this value based on dataset size and desired training time."""

BATCH_SIZE = 16
"""int: Batch size used during training. Adjust based on GPU memory constraints."""

IMG_SIZE = 640
"""int: The size (height and width) of the input images for the model."""

CONFIDENCE_THRESHOLD = 0.25
"""float: The confidence threshold for predictions during validation and testing."""

PROJECT_NAME = 'YOLO11-KITTI'
"""str: The name of the project folder where YOLO results will be saved."""

EXPERIMENT_NAME = 'exp1'
"""str: The name of the experiment folder within the project directory to store this run's results."""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""str: The device to use for training and inference, defaults to GPU if available."""

print(f"Using device: {device}")

# Ensure the expected dataset directories exist
if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
if not LABEL_DIR.exists():
    raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

# Ensure the expected dataset directories exist
if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
if not LABEL_DIR.exists():
    raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

# -----------------------------
# 2. Data Preparation Functions
# -----------------------------

CLAZZ_NUMBERS = {name: idx for idx, name in enumerate(CLASSES)}
"""
dict: A mapping from class names to numeric labels. 
The numeric labels are used by YOLO for class indices.
"""


def convert_bbox_to_yolo(bbox, size):
    """
    Convert KITTI bounding box coordinates to YOLOv11 format.

    Args:
        bbox (tuple of float): Bounding box coordinates in the format (left, right, top, bottom).
        size (tuple of int): Image size as (width, height).

    Returns:
        tuple of float: YOLO-formatted bounding box as (x_center, y_center, width, height) normalized by image size.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (bbox[0] + bbox[1]) / 2.0
    y_center = (bbox[2] + bbox[3]) / 2.0
    width = bbox[1] - bbox[0]
    height = bbox[3] - bbox[2]
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return x_center, y_center, width, height


def parse_kitti_label_file(lbl_path, img_path):
    """
    Parse a KITTI label file and convert the bounding boxes to YOLOv11 format.

    Args:
        lbl_path (Path): Path to the KITTI label file (in KITTI text format).
        img_path (Path): Path to the corresponding image file.

    Returns:
        list of tuple: A list of YOLO-formatted bounding boxes. Each element is 
        (class_idx, x_center, y_center, width, height).
    """
    with open(lbl_path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n')

    yolo_labels = []
    if not img_path.exists():
        # If the image doesn't exist, skip processing labels
        return yolo_labels

    img_size = Image.open(img_path).size  # (width, height)
    
    for line in lines:
        parts = line.split()
        clazz = parts[0]
        if clazz not in CLAZZ_NUMBERS:
            # Skip classes not in our mapping
            continue

        # KITTI format: 
        # type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, ...
        # Indices:  0    ,    1     ,   2     ,   3  ,    4     ,    5    ,     6     ,      7    ...
        # Example: Car 0.00 0 1.57 148.00 174.00 350.00 325.00 ...
        # The bounding box coordinates: left = parts[4], top = parts[5], right = parts[6], bottom = parts[7]
        bbox_left = float(parts[4])
        bbox_top = float(parts[5])
        bbox_right = float(parts[6])
        bbox_bottom = float(parts[7])
        bbox = (bbox_left, bbox_right, bbox_top, bbox_bottom)

        # Convert bounding box to YOLO format (normalized)
        x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_size)
        clazz_number = CLAZZ_NUMBERS[clazz]

        # YOLO format: class x_center y_center width height
        yolo_labels.append((clazz_number, x_center, y_center, width, height))

    return yolo_labels


# -----------------------------
# 3. Generate YOLO Labels
# -----------------------------

if not LABELS_DIR.exists():
    LABELS_DIR.mkdir()

image_paths = sorted(list(IMAGE_DIR.glob('*.png')))
label_paths = sorted(list(LABEL_DIR.glob('*.txt')))

for img_path in image_paths:
    lbl_path = LABEL_DIR / f"{img_path.stem}.txt"
    if lbl_path.exists():
        yolo_labels = parse_kitti_label_file(lbl_path, img_path)
        yolo_label_path = LABELS_DIR / f"{img_path.stem}.txt"
        with open(yolo_label_path, 'w', encoding='utf-8') as lf:
            for lbl in yolo_labels:
                lf.write(" ".join(f"{val:.6f}" for val in lbl) + "\n")

print("YOLO format labels have been generated in:", LABELS_DIR.resolve())

# -----------------------------
# 4. Split Dataset into Train and Validation Sets
# -----------------------------

labels_for_images = [(img_path, LABELS_DIR / f"{img_path.stem}.txt") 
                     for img_path in image_paths 
                     if (LABELS_DIR / f"{img_path.stem}.txt").exists()]

train_pairs, valid_pairs = train_test_split(
    labels_for_images, 
    test_size=0.1, 
    random_state=42, 
    shuffle=True
)
print(f"Training samples: {len(train_pairs)}, Validation samples: {len(valid_pairs)}")

# Create directories for YOLO data structure:
for folder in [TRAIN_DIR, VALID_DIR]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    (folder / 'images').mkdir()
    (folder / 'labels').mkdir()

for img_path, lbl_path in train_pairs:
    shutil.copy(img_path, TRAIN_DIR / 'images' / img_path.name)
    shutil.copy(lbl_path, TRAIN_DIR / 'labels' / lbl_path.name)

for img_path, lbl_path in valid_pairs:
    shutil.copy(img_path, VALID_DIR / 'images' / img_path.name)
    shutil.copy(lbl_path, VALID_DIR / 'labels' / lbl_path.name)

print(f"Training data copied to {TRAIN_DIR / 'images'} and {TRAIN_DIR / 'labels'}")
print(f"Validation data copied to {VALID_DIR / 'images'} and {VALID_DIR / 'labels'}")

# -----------------------------
# 5. Create data.yaml File for YOLO
# -----------------------------

DATA_CONFIG = 'data.yaml'
data_config = {
    'train': str((TRAIN_DIR / 'images').resolve()),
    'val': str((VALID_DIR / 'images').resolve()),
    'names': CLASSES,
    'nc': len(CLASSES)
}

with open(DATA_CONFIG, 'w', encoding='utf-8') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("data.yaml file created with content:")
print(data_config)

# -----------------------------
# 6. Train the YOLOv11 Model
# -----------------------------

model = YOLO(MODEL_ARCH).to(device)
train_results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    device=device,
    exist_ok=True
)

print("\nTraining completed!\n")

# -----------------------------
# 7. Validate the Model
# -----------------------------

best_weights_path = f'{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt'
model = YOLO(best_weights_path).to(device)
validation_results = model.val(
    data=DATA_CONFIG, 
    split='val', 
    conf=CONFIDENCE_THRESHOLD,
    save=False,
    plots=False
)
print("\nValidation completed!\n")

# The result from model.val() is a 'DetMetrics' object that summarizes performance.
# The 'DetMetrics' class holds detection metrics, including mean precision (mp), mean recall (mr), 
# mAP (map50, map50-95), and others.
# Let's inspect the attributes of validation_results for these metrics.

print("Validation Results (raw DetMetrics object):")
print(validation_results)

# According to Ultralytics YOLO code, `validation_results` is a `DetMetrics` object that has a 'box' attribute
# which holds metrics. We can inspect `validation_results.box` as it may contain the summary metrics.
print("Attributes of validation_results:")
for attribute in dir(validation_results):
    if not attribute.startswith('_'):
        print(attribute, "=", getattr(validation_results, attribute))

# Access relevant metrics from the DetMetrics object
metrics = validation_results.box  # box is an instance of the Metric class storing results for boxes
print("\nBox Metrics:")
print(metrics)

# The metrics attribute should include values like 'mp' (mean precision), 'mr' (mean recall), etc.
# According to the docs:
# - metrics.ap50: average precision at IoU=0.50
# - metrics.ap: average precision for IoU=0.50:0.95
# - metrics.mp: mean precision
# - metrics.mr: mean recall
# - metrics.map50: mean average precision at IoU=0.50
# - metrics.map: mean average precision at IoU=0.50:0.95

# We will attempt to retrieve these metrics:
precision = getattr(metrics, 'mp', None)
recall = getattr(metrics, 'mr', None)

# Calculate F1 score if precision and recall are available
f1_score = None
if precision is not None and recall is not None and (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)

# Accuracy is not directly applicable to object detection tasks like YOLO
accuracy = None

# Placeholder for confusion matrix
confusion_matrix = None

# Print metrics in the requested format
print("\nConfusion Matrix:")
if confusion_matrix is not None:
    print(confusion_matrix)
else:
    print("[[ ... ]]")  # Placeholder for an actual confusion matrix if implemented

if accuracy is not None:
    print(f"Accuracy: {accuracy * 100:.2f}%")
else:
    print("Accuracy: Not Applicable for object detection")

if precision is not None:
    print(f"Precision: {precision:.2f}")
else:
    print("Precision: Not Available")

if recall is not None:
    print(f"Recall: {recall:.2f}")
else:
    print("Recall: Not Available")

if f1_score is not None:
    print(f"F1 Score: {f1_score:.2f}")
else:
    print("F1 Score: Not Available")

# -----------------------------
# 8. Predictions on Validation Set (Optional)
# -----------------------------
val_predictions = model.predict(
    source=str((VALID_DIR / 'images').resolve()), 
    save=True, 
    conf=CONFIDENCE_THRESHOLD
)

if val_predictions:
    predictions_save_dir = val_predictions[0].save_dir
    print(f"\nPredictions saved to '{predictions_save_dir}'.\n")
else:
    print("No predictions were made.")
