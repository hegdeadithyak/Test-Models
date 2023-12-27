import ultralytics
from ultralytics import YOLO



model = YOLO('/home/adithya/Downloads/complete.pt')  

# Define path to video file
source = '/home/adithya/Downloads/y2mate.com - Church steeple catches fire collapses after being struck by lightning Shorts_1080p.mp4'
results = model(source, stream=True)  # generator of Results objects
# Load a pretrained YOLOv8n model
model = YOLO('/home/adithya/Downloads/complete.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('/home/adithya/Downloads/y2mate.com - Church steeple catches fire collapses after being struck by lightning Shorts_1080p.mp4', save=True, imgsz=640, conf=0.5)
