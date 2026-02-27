import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Face Attribute Logic Demo", layout="centered")
st.title("🌍 Face Attribute Logic Demo")
st.write("⚠️ This is a rule-based demonstration project. It does NOT determine real nationality.")

# Load emotion model
@st.cache_resource
def load_emotion_model():
    return load_model("face_emotion.h5", compile=False)

emotion_model = load_emotion_model()

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_face_emotion(face_img):
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (48, 48))
    face_gray = face_gray / 255.0
    face_gray = np.reshape(face_gray, (1, 48, 48, 1))

    preds = emotion_model.predict(face_gray, verbose=0)
    return emotion_labels[np.argmax(preds)]

def estimate_skin_tone(face_img):
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    avg_v = np.mean(v)

    if avg_v < 80:
        return "African (Logic-based)"
    elif avg_v < 140:
        return "Indian (Logic-based)"
    else:
        return "USA (Logic-based)"

def detect_dress_color(image, face_box):
    x, y, w, h = face_box
    dress_region = image[y+h:y+2*h, x:x+w]

    if dress_region.size == 0:
        return "Unknown"

    avg_color = np.mean(dress_region.reshape(-1, 3), axis=0)
    b, g, r = avg_color

    if r > g and r > b:
        return "Red"
    elif b > r and b > g:
        return "Blue"
    elif g > r and g > b:
        return "Green"
    else:
        return "Mixed"

uploaded_image = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected")
    else:
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]

        nationality = estimate_skin_tone(face)
        emotion = predict_face_emotion(face)
        dress_color = detect_dress_color(img, (x, y, w, h))

        st.image(img, channels="BGR", caption="Input Image")

        st.subheader("🧾 Output Panel")

        st.write(f"🌍 Estimated Region: {nationality}")
        st.write(f"🎭 Emotion: {emotion}")
        st.write(f"👕 Dress Color: {dress_color}")