import cv2, os
import sys, math, numpy
import face_recognition
import random
from PIL import Image

def maskHalf_landmark(img_dir):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    landmarks = return_landmarks(img)

    pix_range = [(300, -1), (300, -1), (300, -1), (300, -1), (300, -1)]   # (min, max)
    for i in range(0, landmarks['left_eyebrow']):
        pix_range[0] = (min(img[i][:64]), max(img[i][:64]))
    for i in range(landmarks['left_eyebrow'], landmarks['left_eye']):
        pix_range[1] = (min(img[i][:64]), max(img[i][:64]))
    for i in range(landmarks['left_eye'], landmarks['nose_tip']):
        pix_range[2] = (min(img[i][:64]), max(img[i][:64]))
    for i in range(landmarks['nose_tip'], landmarks['chin']):
        pix_range[3] = (min(img[i][:64]), max(img[i][:64]))
    for i in range(landmarks['chin'], 128):
        pix_range[4] = (min(img[i][:64]), max(img[i][:64]))

    for i in range(0, landmarks['left_eyebrow']):
        for j in range(63, 128):
            img[i][j] = random.randint(pix_range[0][0], pix_range[0][1])
            # img[i][j] = 0
    for i in range(landmarks['left_eyebrow'], landmarks['left_eye']):
        for j in range(63, 128):
            img[i][j] = random.randint(pix_range[1][0], pix_range[1][1])
            # img[i][j] = 50
    for i in range(landmarks['left_eye'], landmarks['nose_tip']):
        for j in range(63, 128):
            img[i][j] = random.randint(pix_range[2][0], pix_range[2][1])
            # img[i][j] = 100
    for i in range(landmarks['nose_tip'], landmarks['chin']):
        for j in range(63, 128):
            img[i][j] = random.randint(pix_range[3][0], pix_range[3][1])
            # img[i][j] = 150
    for i in range(landmarks['chin'], 128):
        for j in range(63, 128):
            img[i][j] = random.randint(pix_range[4][0], pix_range[4][1])
            # img[i][j] = 200
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img

def return_landmarks(image):
    npimage = numpy.array(image)
    face_landmarks = face_recognition.face_landmarks(npimage, model='large')[0]

    # save required face landmarks to dictionary
    # left_eyebrow, left_eye, nose_tip, chin
    req_landmarks = {'left_eyebrow':200, 'left_eye':-1, 'nose_tip':-1, 'chin':-1}
    for item in face_landmarks['left_eyebrow']:
        if item[1] < req_landmarks['left_eyebrow']:
            req_landmarks['left_eyebrow'] = item[1]

    for item in face_landmarks['left_eye']:
        if item[1] > req_landmarks['left_eye']:
            req_landmarks['left_eye'] = item[1]

    for item in face_landmarks['nose_tip']:
        if item[1] > req_landmarks['nose_tip']:
            req_landmarks['nose_tip'] = item[1]

    for item in face_landmarks['chin']:
        if item[1] > req_landmarks['chin']:
            req_landmarks['chin'] = item[1]

    # there are 5 segments in the right side of the image
    # top to eyebrows, eyebrows to eyes, eyes to tip of the nose, nose to chin, and below the chin
    return (req_landmarks)


# maskHalf_landmark('/home/soohyeonlee/project/ModifiedFile/split_dataset/aligned_half_mirrored/train/s2_00107_002.png')




# face image
#
# import os
# from PIL import Image, ImageDraw
# import face_recognition
# image_path = 'C:\\Users\\405B\\Desktop\\s2_00005_001.jpg'
# image = face_recognition.load_image_file(image_path)
#
# face_landmarks_list = face_recognition.face_landmarks(image)
#
# pil_image = Image.fromarray(image)
#
# for face_landmarks in face_landmarks_list:
#     for face_landmark in face_landmarks:
#         d = ImageDraw.Draw(pil_image, 'RGB')
#         d.line(face_landmarks[face_landmark], fill='white', width=5)
#
# pil_image.save('C:\\Users\\405B\\Desktop\\s2_00005_001_landmarked.jpg')
