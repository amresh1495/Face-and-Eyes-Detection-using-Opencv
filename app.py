import cv2
import numpy as np
from flask import Flask, render_template, url_for, request

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


app = Flask(__name__)


@app.route('/')
def face_detector():
    capture = cv2.VideoCapture(0)
    while 1:
        ret, img = capture.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            continue
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow('Press ESC to close the Camera', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
    return ("<h2>You have closed the camera...</h2>")


if __name__ == '__main__':
    app.run(debug=True)
