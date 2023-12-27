import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# Load your custom video
video_path = '/home/adithya/Downloads/y2mate.com - Church steeple catches fire collapses after being struck by lightning Shorts_1080p.mp4'
output_path = './video_output.mp4'

# Create a Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for detections

# Load your custom model weights
model_weights_path = '/home/adithya/Downloads/model_final.pth'
cfg.MODEL.WEIGHTS = model_weights_path
cfg.MODEL.DEVICE = "cpu"

# Create a Detectron2 predictor
predictor = DefaultPredictor(cfg)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define a video writer for the output (use mp4 codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'avc1' for mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame in the video
while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if we reached the end of the video

    # Perform inference
    outputs = predictor(frame)

    # Visualize the predictions on the frame
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out_frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Write the frame to the output video
    out.write(out_frame.get_image()[:, :, ::-1])

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

