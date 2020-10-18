
import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eye_image = cv2.imread("eyes.jpg")

    faces = detector(frame)
    for face in faces:

        landmarks = predictor(gray, face)

        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        center_eye = (landmarks.part(27).x, landmarks.part(27).y)
        # left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        # right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        eye_width = int(hypot(left_eye[0] - right_eye[0],
                               left_eye[1] - right_eye[1]) * 1.7)
        eye_height = int(eye_width * 0.6)
        # New nose position
        top_left = (int(center_eye[0] - eye_width / 2),
                    int(center_eye[1] - eye_height / 2))
        bottom_right = (int(center_eye[0] + eye_width / 2),
                        int(center_eye[1] + eye_height / 2))
        eye_new = cv2.resize(eye_image, (eye_width, eye_height))
        # cv2.imshow("nose",eye_new)

        eye_gray = cv2.cvtColor(eye_new, cv2.COLOR_BGR2GRAY)

        _, eye_mask = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("eye1", eye_gray)
        # cv2.imshow("eye2", eye_mask)


        eye_area = frame[top_left[1]: top_left[1] + eye_height,
                    top_left[0]: top_left[0] + eye_width]
        eye_area_no_eye = cv2.bitwise_and(eye_area, eye_area, mask=eye_mask)
        # cv2.imshow("eye2", eye_area_no_eye)

        final_eye = cv2.add(eye_area_no_eye, eye_new)
        frame[top_left[1]: top_left[1] + eye_height,
        top_left[0]: top_left[0] + eye_width] = final_eye
        # cv2.imshow("Nose area", nose_area)
        # cv2.imshow("Nose pig", nose_pig)
        cv2.imshow("final eye", final_eye)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break