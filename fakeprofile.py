import sys
import os
import cv2
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils

from face_recognizer.face_reco_image import FaceImage
from lib.detect_blinks import detect_blink
from lib.face_detect import detect_face
from database.data_factory import dataFactory


class FakeProfilePreventor():
    # global vars
    __consecutive_frames_none = __consecutive_frames_multi = 0
    __video_capture = __face = __data = ""
    __name = __firstname = __lastname = __email_id = __age = __gender = ""
    __key = __blink_count = 0

    # init
    def __init__(self):
        # load and train model
        self.__face = FaceImage()
        self.__data = dataFactory()
        self.__video_start()

    # face detection
    def __face_detection(self, frame):
        face_count, faces = detect_face(frame)

        if (face_count > 1):
            self.__consecutive_frames_multi += 1
            if(self.__consecutive_frames_multi > 10):
                print("\n[Error] More Than One Face Detected")
                cv2.putText(frame, "More Than One Face Detected", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif (face_count < 1):
            self.__consecutive_frames_none += 1
            if(self.__consecutive_frames_none > 10):
                print("\n[Error] No Face Detected")
                cv2.putText(frame, "No Face Detected", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            self.__consecutive_frames_multi = self.__consecutive_frames_none = 0
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        return frame, face_count

    # face recognition

    def __face_recognition(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_id = self.__face.detect_face(rgb)
        if(result_id == "Unknown"):
            result_id = self.__data.registerUser(
                self.__firstname, self.__lastname, self.__age, self.__gender, self.__email_id, "1111")
            if result_id != "Error":
                print("\n[Success] Account Creation Allowed")
                self.__face.create_new_images(
                    frame, self.__name, result_id)
                print("\n[Success] User Facial Data Recorded")
                print("\n[Process] Training Network...")
                self.__face.train_network()
            else:
                print("try again")
        else:
            result_id = self.__data.fakeAccountAttempt(result_id, "DemoOSN")
            print("\n[Failed] Fake Account Creation Attempt Detected")
            print("\n[Failed] Account Already Owned by :", result_id)

        return frame

    # profile data input

    def data_input(self):
        check = True

        while check:
            os.system("cls")
            print("### FAKE PROFILE PREVENTOR ###\n")
            firstname = self.__data.validName(input("Enter First Name : "))
            lastname = self.__data.validName(input("Enter Last Name : "))
            age = self.__data.validNumber(input("Enter age : "))
            print(
                "Choose Gender => \n\t 'M' or 'm' for Male \n\t 'F' or 'f' for Female \n\t 'O' or 'o' for Others")
            gender = self.__data.validGender(input("Enter gender : "))
            email_id = self.__data.validEmail(input("Enter Email Id : "))

            if firstname != False and lastname != False and age != False and gender != False and email_id != False:
                self.__firstname = firstname
                self.__lastname = lastname
                self.__name = firstname + " " + lastname
                self.__age = age
                self.__gender = gender.upper()
                self.__email_id = email_id

                check = False

                print("\n[Process] Starting Face Recognition")
            else:
                print("\n[Error] Data Entered Invalid =>")
                if firstname == False:
                    print("[Error] Firstname Invalid")
                if lastname == False:
                    print("[Error] Lastname Invalid")
                if age == False:
                    print("[Error] Age Invalid")
                if gender == False:
                    print("[Error] Gender Invalid")
                if email_id == False:
                    print("[Error] Email Invalid")
                print("\n#############################################")
                temp = input(
                    "\nPress 'Y' or 'y' to continue or anykey to exit")
                if temp != "Y" or temp != "y":
                    self.code_exit()

    # exit program
    def code_exit(self):
        self.__video_exit()
        sys.exit()

    # video capture
    def __video_start(self):
        self.__video_capture = cv2.VideoCapture(0)
        if not self.__video_capture.isOpened():
            print('Unable to load camera.')
            code_exit()

    # video exit
    def __video_exit(self):
        self.__key = "q"
        self.__video_capture.release()
        cv2.destroyAllWindows()

    # fake profile detector
    def profile_detector(self):
        self.__video_start()

        while not self.__key == ord("q"):
            ret, frame = self.__video_capture.read()
            frame = imutils.resize(frame, width=640)

            # detect number of faces
            if(self.__blink_count < 3):
                frame, face_count = self.__face_detection(frame)

            if(face_count <= 1):

                # detect blinks
                if(self.__blink_count < 3):
                    frame, self.__blink_count = detect_blink(frame)
                else:
                    # face recognition
                    frame = self.__face_recognition(frame)
                    break
            else:
                print("\nBlinks Reset. Try Again")
                self.__blink_count = 0

            if(self.__blink_count <= 3):
                cv2.putText(frame, "Blinks: {}".format(self.__blink_count), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Fake Profile Detection", frame)

            # quit
            key = cv2.waitKey(1) & 0xFF

        self.__video_exit()


if __name__ == "__main__":
    key_input = "y"
    fakeProfile = FakeProfilePreventor()

    while key_input == "Y" or key_input == "y":
        fakeProfile.data_input()
        fakeProfile.profile_detector()
        print("\n ############################################ \n\n")
        key_input = input(
            "Press Y to create another account or any key to exit : ")
    fakeProfile.code_exit()
