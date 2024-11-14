import cv2
import torch
import argparse
import os
from ultralytics import YOLO

def draw_boxes_on_frame(frame, detections, class_names):
    """
    Draws bounding boxes and labels for detected objects on a video frame.

    Args:
        frame (numpy.ndarray): The video frame to draw annotations on.
        detections (list): The detection results from YOLO model for the frame.
        class_names (list): List of class names corresponding to class IDs.
    Returns:
        numpy.ndarray: Annotated frame with bounding boxes and labels drawn.
    """
    for det in detections:
        # YOLO detection format:
        # [x1, y1, x2, y2, confidence, class_id]
        # Where (x1, y1) is top-left corner and (x2, y2) is bottom-right.
        x1, y1, x2, y2, conf, cls = map(float, det)
        label = f"{class_names[int(cls)]} {conf:.2f}"
        # Drawing the bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Drawing the label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        top = int(y1)
        if top - label_size[1] < 0:
            top = label_size[1]
        cv2.rectangle(frame, (int(x1), int(top - label_size[1] - 8)),
                      (int(x1 + label_size[0]), int(top)), (0, 255, 0), cv2.FILLED)
        # Drawing the label text
        cv2.putText(frame, label, (int(x1), int(top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame

def detect_objects_in_video(input_video_path, output_video_path, model, class_names, conf_threshold, iou_threshold, show_live):
    """
    Performs object detection on a video using YOLOv11 model, draws bounding boxes and labels, 
    and saves/display the output.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str): The path to save the annotated output video.
        model (YOLO): The loaded YOLO model.
        class_names (list): List of class names for labels.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).
        show_live (bool): Whether to display the video output in a window.

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video source {input_video_path}")

    # Retrieve video properties for output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # If FPS is zero (e.g., some video files don't specify FPS), set a default
    if fps == 0:
        fps = 30

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video stream
            break

        # Perform object detection with the YOLO model
        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        annotated_frame = frame.copy()

        # Extract detection data
        # YOLO results are in results[0].boxes
        if results and len(results) > 0:
            boxes = results[0].boxes
            # If boxes exist
            if boxes is not None and len(boxes) > 0:
                # Convert these boxes into an array of detections for the draw function:
                detections = []
                for box in boxes:
                    # box.xyxyn format:  [x1, y1, x2, y2, confidence, class_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item() if box.conf is not None else 0
                    cls = box.cls.item() if box.cls is not None else -1
                    # Append to detections
                    detections.append([x1, y1, x2, y2, conf, cls])

                annotated_frame = draw_boxes_on_frame(annotated_frame, detections, class_names)

        # Write the annotated frame to output video
        out.write(annotated_frame)

        # If show_live is True, display the frame
        if show_live:
            cv2.imshow('YOLOv11 Object Detection', annotated_frame)
            # Press 'q' to quit live display
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture and writer, close any open windows
    cap.release()
    out.release()
    if show_live:
        cv2.destroyAllWindows()

    print(f"Object detection on video completed. Output saved to: {output_video_path}")

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Object detection in a video using YOLOv11.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, required=False,
                        help='Path to the output video file. If not specified, `_annotated` will be added to input filename.')
    parser.add_argument('--weights', type=str, default='best.pt', help='Path to the YOLOv11 trained model weights. Default is `best.pt`.')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold for object detection.')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--show-live', action='store_true', help='If set, display the video output in a window during processing.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the YOLO model
    model = YOLO(args.weights).to(device)
    # Check if classes are defined in the model, otherwise define them manually
    class_names = model.names if hasattr(model, 'names') else [
        'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
        'Cyclist', 'Tram', 'Misc', 'DontCare'
    ]

    # Determine output video path
    if args.output:
        output_video_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_video_path = f"{base}_annotated{ext}"

    # Run object detection on video
    detect_objects_in_video(
        input_video_path=args.input,
        output_video_path=output_video_path,
        model=model,
        class_names=class_names,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        show_live=args.show_live
    )

if __name__ == "__main__":
    main()
