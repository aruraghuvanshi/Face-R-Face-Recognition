'''

Face Recognition using AI
Developed by Aru Raghuvanshi

Dataset: Arrange all files in jpg format. The arrangement of each file should
correspond to the serial order of train_names list.

Date: 23-09-2020

'''

import cv2
import face_recognition as fr
import numpy as np
from facetraining import trainAndEncode
from config import datasetpath, verbose, trainflag, train_names
from config import scale, imagepath, mode, dronecam
import pickle
import time
import warnings
from thewhitetello import TheWhiteTello
warnings.filterwarnings('ignore')


def runFaceRecognizer(path=datasetpath, verbose=verbose, mode=mode):

    print('\n\t\t\033[1;31mFACIAL RECOGNITION USING AI\033[0m')
    print('\t\tDeveloped by Aru Raghuvanshi')
    # time.sleep(3)

    if trainflag:
        face_train = trainAndEncode(path)
        with open("face_encodings.txt", "wb") as fp:
            pickle.dump(face_train, fp)
    else:
        print('\n\nLoading \033[1;32mPRE-TRAINED\033[0m Model.')
        with open('face_encodings.txt', 'rb') as fo:
            face_train = pickle.load(fo)

    face_locations = []
    face_names = []
    process = True
    time.sleep(1)

    if mode == 'webcam':
        print('\033[1;33mSTARTING\033[0m Video Stream. Press \033[1;34mESC\033[0m to end program.')
        if dronecam:
            t = TheWhiteTello()
            t.connect()
            t.streamon()
            cap = cv2.VideoCapture('udp://@0.0.0.0:11111')
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        while True:
            ret, frame = cap.read()
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small_frame = small_frame[:, :, ::-1]

            if process:
                face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = fr.compare_faces(face_train, face_encoding, tolerance=0.6)
                    name = "Not Recognised"
                    face_distances = fr.face_distance(face_train, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = train_names[best_match_index]
                    face_names.append(name)

            process = not process
            if verbose: print(f'Face Detected - {face_names}')
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 1
                right *= 1
                bottom *= 1
                left *= 1
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # cv2.rectangle(frame, (left, bottom - 10), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left+6, bottom-12), font, 0.7, (255, 255, 255), 1)
                # cv2.putText(frame, '-- AROX.AI', (left+6, bottom-4), font, 0.3, (255, 0, 0), 1)

            cv2.imshow('Video', frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27:  # press 'ESC' to quit
                break
        cap.release()
        print('--PROGRAM \033[1;31mENDED\033[0m')

    elif mode == 'image':
        frame = cv2.imread(imagepath)
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = small_frame[:, :, ::-1]
        # rgb_small_frame = np.array(rgb_small_frame)
        frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)

        if process:
            face_locations = fr.face_locations(rgb_small_frame)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = fr.compare_faces(face_train, face_encoding, tolerance=0.6)
                name = "Not Recognised"
                face_distances = fr.face_distance(face_train, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = train_names[best_match_index]
                face_names.append(name)

        # process = not process
        if verbose: print(f'Face Detected - {face_names}')
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 1
            right *= 1
            bottom *= 1
            left *= 1

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # cv2.rectangle(frame, (left, bottom - 10), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 12), font, 0.7, (255, 255, 255), 1)
            # cv2.putText(frame, '-- AROX.AI', (left+6, bottom-4), font, 0.3, (255, 0, 0), 1)


        cv2.imshow('image', frame)
        cv2.waitKey(0)
        print('--PROGRAM \033[1;31mENDED\033[0m')


    cv2.destroyAllWindows()
    return


runFaceRecognizer(datasetpath, verbose)