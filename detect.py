import cv2
from ultralytics import YOLO

def detect_signs_in_video(video_path, model_path, output_path=None):
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize video writer if output_path is provided
    out = None
    if output_path:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Annotate frame with results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
                conf = box.conf[0]  # Confidence
                cls = int(box.cls[0])  # Class index
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLO Detection", frame)

        # Save frame to output video if writer is initialized
        if out:
            out.write(frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


# Path to your video, model, and output
video_path = "C:\\Users\\ASUS\\OneDrive\\Desktop\\ISL.mp4"
 # Replace with your video file
model_path = "model\\best.pt"  # Replace with your YOLO model file
output_path = "output_video.mp4"  # Replace with desired output file name, or None

# Run detection
detect_signs_in_video(video_path, model_path, output_path)
