# src/main.py
import cv2
from detection import load_model, detect_objects
from tracking import initialize_tracker

def main():
    # Load YOLO model and class labels
    net, classes, output_layers = load_model()

    # Choose the input type (image, video, or webcam)
    input_type = 'video'  # Options: 'image', 'video', or 'webcam'

    if input_type == 'image':
        # Load an image from disk
        image = cv2.imread(r"D:\image-captioning\dataset\Images\image224.jpg")
        if image is None:
            print("Image not found")
            return
        # Perform detection and tracking on the image
        boxes, confidences, class_ids = detect_objects(image, net, classes, output_layers)
        # Draw bounding boxes on the image
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the image with detections
        cv2.imshow("Object Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif input_type == 'video':
        # Load a video file
        cap = cv2.VideoCapture(r"C:\Users\Tanmay\Downloads\Slavia.mp4")
        if not cap.isOpened():
            print("Error: Could not open video")
            return
        # Initialize the tracker
        tracker = initialize_tracker("CSRT")
        tracking_objects = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            boxes, confidences, class_ids = detect_objects(frame, net, classes, output_layers)

            # Perform tracking (for each detected object)
            for i, (x, y, w, h) in enumerate(boxes):
                if len(tracking_objects) <= i:  # Initialize tracker for new object
                    tracker.init(frame, (x, y, w, h))
                    tracking_objects.append((tracker, (x, y, w, h)))
                else:  # Update existing tracker
                    tracker, _ = tracking_objects[i]
                    success, new_box = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in new_box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw detection boxes
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow("Object Tracking", frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif input_type == 'webcam':
        # Use the webcam for real-time video feed
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        # Initialize the tracker
        tracker = initialize_tracker("CSRT")
        tracking_objects = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            boxes, confidences, class_ids = detect_objects(frame, net, classes, output_layers)

            # Perform tracking (for each detected object)
            for i, (x, y, w, h) in enumerate(boxes):
                if len(tracking_objects) <= i:  # Initialize tracker for new object
                    tracker.init(frame, (x, y, w, h))
                    tracking_objects.append((tracker, (x, y, w, h)))
                else:  # Update existing tracker
                    tracker, _ = tracking_objects[i]
                    success, new_box = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in new_box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw detection boxes
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow("Object Tracking", frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
