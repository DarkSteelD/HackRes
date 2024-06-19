import cv2
import os
from ultralytics import YOLO

def log_message(message):
    print(f"[INFO] {message}")

# List of model weights
model_weights = [
    os.path.expanduser('~/Documents/YOLOMLSHIT/CIz.pt'),
    os.path.expanduser('~/Documents/YOLOMLSHIT/FireSmoke.pt'),
    os.path.expanduser('~/Documents/YOLOMLSHIT/Smoking.pt'),
    os.path.expanduser('~/Documents/YOLOMLSHIT/toolsdetect.pt'),
    os.path.expanduser('~/Documents/YOLOMLSHIT/transport.pt')
]

log_message("Loading models...")
# Load the YOLOv8 models and their class names
models = []
for weight in model_weights:
    try:
        model = YOLO(weight)
        class_names = model.names
        models.append((model, class_names))
        log_message(f"Model loaded: {weight}")
    except Exception as e:
        log_message(f"Error loading model {weight}: {e}")

# Path to the input video file or set to 0 for webcam
video_path = os.path.expanduser('~/Documents/YOLOMLSHIT/123.mp4')

# Initialize the video capture from the specified video file or webcam
log_message(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    log_message("Error: Unable to open video file.")
    exit(1)

# Define the codec and create VideoWriter object to save the output video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'H264' for H.264 codec
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

confidence_threshold = 0.65

log_message("Starting video processing...")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        log_message("End of video or unable to read frame.")
        break
    
    combined_results = []
    
    # Perform detection using each model
    for model, class_names in models:
        try:
            results = model(frame)  # Get predictions from the model
            if len(results) > 0:
                boxes = results[0].boxes.data.tolist()  # Get the bounding boxes and other details
                
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box[:6]
                    if conf >= confidence_threshold:
                        combined_results.append((int(x1), int(y1), int(x2), int(y2), conf, int(cls), class_names[int(cls)]))  # Add class names to results
        except Exception as e:
            log_message(f"Error processing model {model}: {e}")
    
    # Draw results on the frame
    for x1, y1, x2, y2, conf, cls, label in combined_results:
        color = (0, 255, 0) if conf > 0.7 else (0, 0, 255)  # Green for high confidence, red for lower
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Save the frame with predictions to the output video
    out.write(frame)
    
    # Display the frame with predictions
    cv2.imshow('Video - YOLOv8 Predictions', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

log_message("Processing complete.")
