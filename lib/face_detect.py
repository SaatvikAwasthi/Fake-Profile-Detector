"""
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "./models/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def detect_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return len(faces), faces
"""

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "./models/shape_predictor_68_face_landmarks.dat")


def detect_face(frame):
    ls = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        ls.append([x, y, w, h])

        # for (x, y) in shape:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    return len(ls), ls
