import cv2
import os
import numpy as np
import dlib
import copy

from face_recognizer.face_reco_base import FaceRecognizer
from lib.face_detect import detect_face


class FaceImage(object):

    face_recognizer = ""

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceImage, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        os.system("cls")
        print("Loading face recognizer...")
        self.face_recognizer = FaceRecognizer()
        print("Loaded face recognizer")
        input("Press any key to continue...")

    def detect_face(self, img):
        img = copy.deepcopy(img)

        # for face detection
        detector = dlib.get_frontal_face_detector()

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        face_bbs, identities = self.face_recognizer.identify_image_faces(
            img)

        return identities[0]

    def train_network(self):
        self.face_recognizer.train_images()

    def __create_folder(self, folder_name):
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    def create_new_images(self, frame, name, email_id):
        face_dir = "faces/"
        counter = 1

        self.__create_folder(face_dir)

        face_folder = "./" + face_dir + str(email_id) + "/"
        self.__create_folder(face_folder)

        while(counter <= 10):
            face_count, faces = detect_face(frame)
            if(face_count == 1):
                img_path = face_folder + \
                    str(name) + "_" + str(counter) + ".jpg"
                cv2.imwrite(img_path, frame)
                counter += 1
                #print("Saved", img_path)
