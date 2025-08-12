# import cv2 
# import numpy as np
# from wavelet import w2d
# import json
# import joblib
# import base64

# __class_name_to_number = {}
# __class_number_to_name = {}

# __model = None

# def classify_image(image_base64_data, file_path=None):
#     imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
#     result = []

#     for img in imgs:
#         scalled_raw_img = cv2.resize(img, (32, 32))
#         im_har = w2d(scalled_raw_img, mode='haar', level=1)
#         scaled_img_har = cv2.resize(im_har, (32, 32))
#         combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scaled_img_har.reshape(32*32, 1)))
        
#         len_img_array = 32*32*3 + 32*32
#         final_input = combined_img.reshape(1, len_img_array)

#         result.append({
#             'class': __class_number_to_name[__model.predict(final_input)[0]],  # fixed dictionary access here
#             'class_probability': np.around(__model.predict_proba(final_input)*100, 2).tolist()[0],
#             'class_dictionary': __class_name_to_number
#         })
#     return result

# def class_number_to_name(class_num):
#     return __class_number_to_name[class_num]

# def load_saved_artifacts():
#     print("Loading saved artifacts ...")
#     global __class_name_to_number
#     global __class_number_to_name
#     global __model

#     with open("./artifacts/class_dictionary.json", "r") as f:
#         __class_name_to_number = json.load(f)
#         __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

#     if __model is None:
#         with open('./artifacts/saved_model.pkl', 'rb') as f:
#             __model = joblib.load(f)
#     print("loading saved artifacts...done")

# def get_cv2_image_from_base64_string(b64str):
#     '''
#     Credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
#     '''
#     encoded_data = b64str.split(',')[1]
#     nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img

# def get_cropped_image_if_2_eyes(image_path, image_base64_data):
#     face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
#     eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

#     if image_path:
#         img = cv2.imread(image_path)
#     else:
#         img = get_cv2_image_from_base64_string(image_base64_data)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     cropped_faces = []
#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         if len(eyes) >= 2:
#             cropped_faces.append(roi_color)
#     return cropped_faces

# def get_b64_test_image_for_virat():
#     with open("img_b64.txt") as f:
#         return f.read()

# if __name__ == "__main__":
#     load_saved_artifacts()
#     print(classify_image(get_b64_test_image_for_virat(), None))



import cv2 
import numpy as np
from wavelet import w2d
import json
import joblib

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
            (scalled_raw_img.reshape(32*32*3, 1), scaled_img_har.reshape(32*32, 1))
        )

        len_img_array = 32*32*3 + 32*32
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

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)

def get_cropped_image_if_2_eyes_from_bytes(image_bytes):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

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
        elif len(eyes) == 1 and not cropped_faces:  # fallback if no 2-eye detected yet
            cropped_faces.append(roi_color)
        elif not cropped_faces:  # final fallback: at least return the face
            cropped_faces.append(roi_color)

    return cropped_faces
