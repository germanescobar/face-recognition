"""
This script is used to train the face recognizer using positive and negative
images. The output is an XML file with name 'training-data.xml'
"""

import os
import cv2
import cv
import numpy as np

def load_images(images_dir):
    all_images = os.listdir(images_dir)
    for image_name in all_images:
        full_path = images_dir + image_name
        print full_path
        try:
            image = cv.LoadImage(full_path, cv2.IMREAD_GRAYSCALE)
        except IOError:
            continue
        yield image[:, :]


if __name__ == "__main__":
    recognizer = cv2.createLBPHFaceRecognizer()

    training_images = []
    training_labels = []
    for image in load_images("positive/"):
        training_images.append(np.asarray(image));
        training_labels.append("German") # change with your name


    for image in load_images("negative/"):
        training_images.append(np.asarray(image));
        training_labels.append(0)

    recognizer.train(np.asarray(training_images), np.asarray(training_labels)) 
    recognizer.save('training-data.xml')
        