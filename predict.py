from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import imutils
from mtcnn.mtcnn import MTCNN
import datetime
face_classifier = MTCNN()
classifier =load_model(r'/REPORT/1/MODEL/model.h5')
emotion_labels = ['real','fake']

cap = cv2.VideoCapture(0)
total_frames=0
while True:
    _, frame = cap.read()
    frame=imutils.resize(frame,width=800)
    total_frames = total_frames + 1
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = face_classifier.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        x2, y2 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(96, 96),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            prediction = classifier.predict(roi)
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y-5)

            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('face detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()