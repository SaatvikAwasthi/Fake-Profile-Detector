
from sklearn.manifold import TSNE
import imageio
from face_recognizer.utils import display_cv2_image
import warnings
import copy
import time
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from face_recognizer.utils import load_image
from face_recognizer.align import AlignDlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os.path
import numpy as np
from tensorflow import set_random_seed
import dlib
import cv2
import bz2
import os
from face_recognizer.model import create_model

from urllib.request import urlopen

from numpy.random import seed
seed(1)
set_random_seed(2)


# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')
MIN_DLIB_SCORE = 1.1
MIN_SHARPNESS_LEVEL = 30
MIN_CONFIDENCE_SCORE = 0.3


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


class FaceRecognizer():
    def __init__(self):
        dst_dir = './models'
        dst_file = os.path.join(
            dst_dir, 'shape_predictor_68_face_landmarks.dat')

        # Create CNN model and load pretrained weights (OpenFace nn4.small2)
        self.nn4_small2_pretrained = create_model()
        self.nn4_small2_pretrained.load_weights('./models/nn4.small2.v1.h5')
        self.metadata = self.load_metadata('./faces')

        # Initialize the OpenFace face alignment utility
        self.alignment = AlignDlib(
            './models/shape_predictor_68_face_landmarks.dat')

        # Get embedding vectorsf
        # self.embedded = np.zeros((self.metadata.shape[0], 128))
        self.embedded = np.zeros((0, 128))

        # Train images
        print("Training Images")
        custom_metadata = self.load_metadata("./faces")
        self.metadata = np.append(self.metadata, custom_metadata)
        self.update_embeddings()
        # self.visualize_dataset()
        self.train_images()

    def load_metadata(self, path):
        ds_store = ".DS_Store"
        metadata = []
        dirs = os.listdir(path)
        if ds_store in dirs:
            dirs.remove(ds_store)
        for i in dirs:
            subdirs = os.listdir(os.path.join(path, i))
            if ds_store in subdirs:
                subdirs.remove(ds_store)
            for f in subdirs:
                metadata.append(IdentityMetadata(path, i, f))
        return np.array(metadata)

    # Align helper functions

    def get_face_thumbnail(self, img):
        return self.alignment.getLargestFaceThumbnail(96, img, self.alignment.getLargestFaceBoundingBox(img),
                                                      landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    def get_all_face_thumbnails_and_scores(self, img):
        return self.alignment.getAllFaceThumbnailsAndScores(96, img,
                                                            landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    def get_face_vector(self, img, is_thumbnail=False):
        if not is_thumbnail:
            img = self.get_face_thumbnail(img)
        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
        # obtain embedding vector for image
        return self.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

    def get_face_vectors(self, img):
        face_thumbnails, scores, face_types = self.get_all_face_thumbnails_and_scores(
            img)
        face_vectors = []
        for face_img in face_thumbnails:
            # scale RGB values to interval [0,1]
            face_img = (face_img / 255.).astype(np.float32)
            # obtain embedding vector for image
            vector = self.nn4_small2_pretrained.predict(
                np.expand_dims(face_img, axis=0))[0]
            face_vectors.append(vector)
        return face_vectors, face_thumbnails, scores, face_types

    # Train classifier models

    def train_images(self, train_with_all_samples=False):
        self.targets = np.array([m.name for m in self.metadata])
        start = time.time()

        self.encoder = LabelEncoder()
        self.encoder.fit(self.targets)

        # Numerical encoding of identities
        y = self.encoder.transform(self.targets)

        if train_with_all_samples == False:
            train_idx = np.arange(self.metadata.shape[0]) % 2 != 0
        else:
            train_idx = np.full(self.metadata.shape[0], True)

        self.test_idx = np.arange(self.metadata.shape[0]) % 2 == 0

        # 50 train examples of 10 identities (5 examples each)
        X_train = self.embedded[train_idx]
        # 50 test examples of 10 identities (5 examples each)
        X_test = self.embedded[self.test_idx]

        y_train = y[train_idx]
        y_test = y[self.test_idx]

        self.knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        self.svc = LinearSVC()  # class_weight='balanced')

        self.knn.fit(X_train, y_train)
        self.svc.fit(X_train, y_train)

        acc_knn = accuracy_score(y_test, self.knn.predict(X_test))
        acc_svc = accuracy_score(y_test, self.svc.predict(X_test))

        if train_with_all_samples == False:
            print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')
        else:
            print('Trained classification models with all image samples')

        end = time.time()
        print("train_images took {} secs".format(end-start))

    def update_embeddings(self):
        for i, m in enumerate(self.metadata):
            #print("loading image from {}".format(m.image_path()))
            img = load_image(m.image_path())
            is_thumbnail = "customer_" in m.image_path()
            vector = self.get_face_vector(img, is_thumbnail)
            vector = vector.reshape(1, 128)
            self.embedded = np.append(self.embedded, vector, axis=0)

    def label_cv2_image_faces(self, rgb_img, face_bbs, identities):
        # Convert RGB back to cv2 RBG format
        img = rgb_img[:, :, ::-1]

        for i, bb in enumerate(face_bbs):
            # Draw bounding rectangle around face
            cv2.rectangle(img, (bb.left(), bb.top()),
                          (bb.right(), bb.bottom()), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (bb.left(), bb.bottom() - 35),
                          (bb.right(), bb.bottom()), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, identities[i], (bb.left(
            ) + 6, bb.bottom() - 6), font, 1.0, (255, 255, 255), 1)
        return img

    def get_sharpness_level(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(grey, cv2.CV_64F).var()

    def identify_image_faces(self, example_image):
        vectors, thumbnails, dlib_scores, face_types = self.get_face_vectors(
            example_image)

        identities = []
        saved_unknown = False
        for i, vector in enumerate(vectors):
            vector = vector.reshape(1, 128)
            confidence_scores = self.svc.decision_function(vector)
            if (confidence_scores.max() < MIN_CONFIDENCE_SCORE):
                sharpness_level = self.get_sharpness_level(thumbnails[i])
                example_identity = "Unknown"
                # example_identity = "Unknown ({:0.2f}, {}, {:0.2f})".format(dlib_scores[i], face_types[i], sharpness_level)
                # print("Unknown face - dlib score={:0.2f}, face_type={}, sharpness_level={:0.2f}".format(
                #    dlib_scores[i], face_types[i], sharpness_level))
            else:
                example_prediction = self.svc.predict(vector)
                example_identity = self.encoder.inverse_transform(example_prediction)[
                    0]
            identities.append(example_identity)

        # Detect faces and return bounding boxes
        face_bbs = self.alignment.getAllFaceBoundingBoxes(example_image)

        return face_bbs, identities

    def visualize_dataset(self):
        X_embedded = TSNE(n_components=2).fit_transform(self.embedded)
        plt.figure()

        for i, t in enumerate(set(self.targets)):
            idx = self.targets == t
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

        plt.legend(bbox_to_anchor=(1, 1))
