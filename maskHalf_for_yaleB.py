from PIL import Image
import face_recognition
import maskHalf_with_landmark, maskHalf
import random
import os, cv2

def find_landmark_each_subject(dir):
    landmark_for_subject = {}  # {'f[4:7]': {left_eyebrow:, left_eye:, nose_tip:, chin:},[left_eyebrow, left_eye, nose_tip, chin] ... }
    for dirpath, dirnames, filenames in os.walk(dir):
        filestoCheck = []
        for f in filenames:
            if 'P00A' in f:
                filestoCheck.append(f);
        for f in filestoCheck:
            if 'P00A+000E+00' in f:
                try:
                    # landmark 찾기
                    image = face_recognition.load_image_file(dirpath + '/' + f)
                    face_recognition.face_landmarks(image, model='large')[0]
                    landmark_for_subject[f[4:7]] = maskHalf_with_landmark.return_landmarks(image)

                except Exception as e:
                    print(e)
    return landmark_for_subject

# maskHalf for yale b database
def maskHalf_for_yaleB_save(img_dir):
    landmark_for_subject = find_landmark_each_subject(img_dir)  # {'f[4:7]': {left_eyebrow:, left_eye:, nose_tip:, chin:},[left_eyebrow, left_eye, nose_tip, chin] ... }
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for f in filenames:
            image_path = os.path.join(dirpath, f)
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                landmarks = landmark_for_subject[f[4:7]]

                pix_range = [(300, -1), (300, -1), (300, -1), (300, -1), (300, -1)]  # (min, max)
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

                # img = halfCopy(image_path)
                # img = pushRightHalfCopy(image_path, 4)
                # img = maskHalf_with_landmark.maskHalf_landmark(image_path)
                img = maskHalf.blurCenter(img)
                cv2.imwrite(f[:-3] + 'png', img)
            except Exception as e:
                print(e)
                print(image_path, 'ignored')


maskHalf_for_yaleB_save('./')