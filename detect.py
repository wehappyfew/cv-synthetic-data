from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./runs/detect/train4/weights/best.pt")
# model = YOLO("yolo_models/yolo11n.pt") # debug

# Define path to directory containing images and videos for inference
source = "./assets/synthetic_data/images/test/lancet_test.png"

# Run inference on the source
results = model(source, conf=0.5)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
