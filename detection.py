# src/detection.py
import cv2
import numpy as np

def load_model():
    # Load YOLO model
    net = cv2.dnn.readNet("data/yolov3.weights", "data/yolov3.cfg")
    with open("data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()
    
    # Handle potential scalar or array return
    if isinstance(out_layer_indices, list):
        output_layers = [layer_names[i[0] - 1] for i in out_layer_indices]
    else:
        output_layers = [layer_names[i - 1] for i in out_layer_indices]

    return net, classes, output_layers

def detect_objects(image, net, classes, output_layers):
    # Prepare the image for YOLO
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize variables to hold detection information
    class_ids = []
    confidences = []
    boxes = []

    # Process the detection results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids
