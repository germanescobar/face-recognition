"""
This script is used to capture an image and recognize a face using the data
generate by the 'train.py' script.
"""

import cv
import cv2
import sys
import numpy as np
from subprocess import call

def normalize_face_size(face):
    normalized_face_dimensions = (100, 100)
    face_as_array = np.asarray(face)
    resized_face = cv2.resize(face_as_array, normalized_face_dimensions)
    resized_face = cv.fromarray(resized_face)
    return resized_face


def normalize_face_histogram(face):
    face_as_array = np.asarray(face)
    equalized_face = cv2.equalizeHist(face_as_array)
    equalized_face = cv.fromarray(equalized_face)
    return equalized_face


def normalize_face_color(face):
    gray_face = cv.CreateImage((face.width, face.height), 8, 1)
    if face.channels > 1:
        cv.CvtColor(face, gray_face, cv.CV_BGR2GRAY)
    else:
        # image is already grayscale
        gray_face = cv.CloneMat(face[:, :])
    return gray_face[:, :]


def normalize_face_for_save(face):
    face = normalize_face_size(face)
    face = normalize_face_color(face)
    face = normalize_face_histogram(face)
    return face

def capture_image():
    vidcap = cv2.VideoCapture()
    vidcap.open(0)
    retval, image = vidcap.retrieve()
    vidcap.release()
    return image

def identify_faces(image):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print "Found {0} faces!".format(len(faces))
    return faces

def predict_faces(recognizer, image, faces):
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = normalize_face_for_save(face)

        label, distance = recognizer.predict(np.asarray(face))
        if (label != 0):
            call(["say", "-v", "Vicki", "Hi %s, welcome back" % label])

if __name__ == "__main__":
    recognizer = cv2.createLBPHFaceRecognizer()
    recognizer.load("training-data.xml");

    image = capture_image()
    faces = identify_faces(image)
    predict_faces(recognizer, image, faces)

    # this is just to see the faces that it used
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)