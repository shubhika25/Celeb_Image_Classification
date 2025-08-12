import cv2
import numpy as np
from wavelet import w2d
import json
import joblib
import os

# Base directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def classify_image_bytes(image_bytes):
    imgs = get_cropped_image_if_2_eyes_from_bytes(image_bytes)
    result = []

    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        im_har = w2d(scalled_raw_img, mode='haar', level=1)
        scaled_img_har = cv2.resize(im_har, (32, 32))
        combined_img = np.vstack(
            (scalled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1))
        )

        len_img_array = 32 * 32 * 3 + 32 * 32
        final_input = combined_img.reshape(1, len_img_array)

        prediction = __model.predict(final_input)[0]
        prediction_proba = __model.predict_proba(final_input)[0]

        result.append({
            'class': __class_number_to_name[prediction],
            'class_probability': np.around(prediction_proba * 100, 2).tolist(),
            'class_dictionary': __class_name_to_number
        })

    return result


def load_saved_artifacts():
    global __class_name_to_number
    global __class_number_to_name
    global __model

    # Load class mapping
    with open(os.path.join(BASE_DIR, "artifacts", "class_dictionary.json"), "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    # Load model
    if __model is None:
        with open(os.path.join(BASE_DIR, "artifacts", "saved_model.pkl"), 'rb') as f:
            __model = joblib.load(f)


def get_cropped_image_if_2_eyes_from_bytes(image_bytes):
    face_cascade = cv2.CascadeClassifier(
        os.path.join(BASE_DIR, "opencv", "haarcascades", "haarcascade_frontalface_default.xml")
    )
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(BASE_DIR, "opencv", "haarcascades", "haarcascade_eye.xml")
    )

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return []  # invalid image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Rule: Prefer 2+ eyes, else allow 1, else fallback to face
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
        elif len(eyes) == 1 and not cropped_faces:
            cropped_faces.append(roi_color)
        elif not cropped_faces:
            cropped_faces.append(roi_color)

    return cropped_faces
