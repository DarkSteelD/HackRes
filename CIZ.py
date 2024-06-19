import cv2
import os
import json
from ultralytics import YOLO

# Expand the path to the best.pt file
model_path = os.path.expanduser('~/Documents/YOLOMLSHIT/CIz.pt')

# Load the YOLOv8 model
model = YOLO(model_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)

all_labels = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Make predictions
    results = model.predict(source=frame, show=True)

    # Extract labels from the results
    boxes = results[0].boxes.data.tolist()  # List of detections
    labels = [model.names[int(box[5])] for box in boxes]  # Assuming class index is at index 5
    all_labels.extend(labels)
    
    # Display the frame with predictions
    cv2.imshow('Webcam - YOLOv8 Predictions', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Convert all labels to JSON and print
all_labels_json = json.dumps(all_labels)
print(all_labels_json)
