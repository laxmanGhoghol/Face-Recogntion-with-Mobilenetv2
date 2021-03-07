import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

a = {0: 'kaalu', 1: 'lakha', 2: 'riya', 3: 'unknown'}


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations and list of predictions

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            # we need the X,Y coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face = frame[startY:endY, startX:endX]
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

        # only make a predictions if atleast one face was detected
        if len(faces) > 0:
            faces = np.array(faces, dtype='float32')
            preds = maskNet.predict(faces, batch_size=4)

        return (locs, preds)


prototxtPath = os.path.sep.join(['deploy.prototxt'])
weightsPath = os.path.sep.join(['res10_300x300_ssd_iter_140000.caffemodel'])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model('mobilenet_v2.model')
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    (locs, preds) = detect_and_predict_mask(frame, faceNet, model)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        val = np.argmax(pred)
        if val < 0.5:
            continue
        print(pred)
        cv2.putText(frame, a[val], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
