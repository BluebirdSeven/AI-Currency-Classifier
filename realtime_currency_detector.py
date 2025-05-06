import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your model path if needed

# Load webcam
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if you have multiple cameras

# Define class names
class_names = ['Fifty', 'Five', 'Hundred', 'One', 'Ten', 'Twenty', 'Two']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, imgsz=640, conf=0.5)[0]

    # Draw results
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Dollar Classifier", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
