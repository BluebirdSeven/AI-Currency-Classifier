import cv2
import threading
import time
import pyttsx3
from ultralytics import YOLO
from tkinter import *
from PIL import Image, ImageTk

# Load model
model = YOLO("best.pt")

# Classes
class_names = ['Fifty', 'Five', 'Hundred', 'One', 'Ten', 'Twenty', 'Two']

# Initialize TTS
engine = pyttsx3.init()
tts_enabled = True

# Globals
# GUI setup
root = Tk()
root.title("Currency Classifier")

last_spoken = 0
speak_interval = 2  # seconds
top_label = StringVar()
top_label.set("Label: N/A")

# Video capture
cap = cv2.VideoCapture(0)

def toggle_tts():
    global tts_enabled
    tts_enabled = not tts_enabled
    tts_button.config(text="TTS: ON" if tts_enabled else "TTS: OFF")

def update_frame():
    global last_spoken

    ret, frame = cap.read()
    if not ret:
        return

    results = model(frame, imgsz=640, conf=0.5)[0]

    highest_conf = 0
    prominent = None

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{class_names[cls_id]} {conf:.2f}"

        if conf > highest_conf:
            highest_conf = conf
            prominent = class_names[cls_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Text-to-speech logic
    if prominent:
        top_label.set(f"Label: {prominent}")
        current_time = time.time()
        if tts_enabled and current_time - last_spoken > speak_interval:
            engine.say(prominent)
            engine.runAndWait()
            last_spoken = current_time
    else:
        top_label.set("Label: N/A")

    # Convert frame to Tkinter format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# GUI setup
video_label = Label(root)
video_label.pack()

label_display = Label(root, textvariable=top_label, font=("Helvetica", 18))
label_display.pack(pady=5)

tts_button = Button(root, text="TTS: ON", command=toggle_tts, font=("Helvetica", 14))
tts_button.pack(pady=5)

# Start update loop in a thread
update_thread = threading.Thread(target=update_frame)
update_thread.daemon = True
update_thread.start()

# Run GUI
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
