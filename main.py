import os
import cv2
import pytesseract
from PIL import Image
from ultralytics import YOLO
import pyttsx3
import time

# ✅ Set path to Tesseract executable (Update if yours is different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Change this to any image name you have in the same folder
image_path = "test_image.jpg" #image type

# ✅ Check if file exists
if not os.path.isfile(image_path):
    print(f"\n❌ ERROR: File '{image_path}' not found.")
    exit()

# ✅ Load image using OpenCV
img = cv2.imread(image_path)
if img is None:
    print(f"\n❌ ERROR: Unable to load image. Check the file format or path.")
    exit()

# ---------- TEXT RECOGNITION ----------
print("\n🔤 Extracted Text:")
text = ""
try:
    text = pytesseract.image_to_string(img).strip()
    print(text if text else "(No text found)")
except Exception as e:
    print(f"❌ OCR Error: {e}")

# ---------- OBJECT DETECTION ----------
print("\n🧠 Detected Objects:")
detected_objects = []
try:
    model = YOLO("yolov8n.pt")
    results = model(image_path, conf=0.4)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = results[0].names[cls]
        detected_objects.append(name)
        print(f"→ {name}")

    if not detected_objects:
        print("(No objects detected)")

    results[0].show()

except Exception as e:
    print(f"❌ YOLO Error: {e}")

# ---------- GATHER TEXT & OBJECTS ----------
speech_parts = []

if text:
    speech_parts.append(f"The extracted text from the image says: {text}")
else:
    speech_parts.append("No text was found in the image.")

if detected_objects:
    object_list = ", ".join(detected_objects)
    speech_parts.append(f"The objects detected in the image are: {object_list}")
else:
    speech_parts.append("No objects were detected in the image.")

# ---------- VOICE OUTPUT ----------
def speak_text_english(message):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 135)
        engine.say("Here is the analysis of the image.")
        engine.runAndWait()
        time.sleep(0.5)
        engine.say(message)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"❌ Voice Error: {e}")

# Join everything into one string to avoid delay
full_message = ". ".join(speech_parts)
print(f"🔊 Speaking: {full_message}")
speak_text_english(full_message)


