import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

model = load_model("C:\\JUNHA\\model.h5", custom_objects={"Adam": Adam})

my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

cap = cv2.VideoCapture(0)  # 0 for the default camera
ret, frame = cap.read()
# dsadsad
img_size = (224, 224)
img = cv2.resize(frame, img_size)  # resize the image to the desired size
img = img / 255.0  # normalize the pixel values to be between 0 and 1
img = np.expand_dims(img, axis=0)

preds = model.predict(img)

left_eye_x1, left_eye_y1, left_eye_x2, left_eye_y2 = preds[0][
    0:4
]  # extract the coordinates for the left eye
right_eye_x1, right_eye_y1, right_eye_x2, right_eye_y2 = preds[0][
    4:8
]  # extract the coordinates for the right eye
frame = cv2.rectangle(
    frame, (left_eye_x1, left_eye_y1), (left_eye_x2, left_eye_y2), (0, 255, 0), 2
)  # draw a green rectangle around the left eye
frame = cv2.rectangle(
    frame, (right_eye_x1, right_eye_y1), (right_eye_x2, right_eye_y2), (0, 255, 0), 2
)  # draw a green rectangle around the right eye

cv2.imshow("Drowsiness Detection", frame)
cv2.waitKey(0)  # wait for a key press
