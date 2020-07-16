import os, cv2, shutil, face_recognition, statistics, random
from PIL import Image
import face_recognition
from PrintException import PrintException
from faceAlign import ReturnCropXYSize
from faceAlign import ModifiedCropFace2 as CropFace


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#          the yale extended b database 정렬 및 파일 정리          #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# only applies to yale extended b database
def alignAndCutYale(angle_name, dir, save_dir):
    eye_landmarks = {}
    crop_xy = {}
    crop_size = {}
    for dirpath, dirnames, filenames in os.walk(dir):
        print('dirpath: ' + dirpath)
        filestoCheck = []
        for f in filenames:
            if 'P00A' in f:
                filestoCheck.append(f);
        for f in filestoCheck:
            if 'P00A+000E+00' in f:
                try:
                    print(f)
                    # landmark 찾기
                    image = face_recognition.load_image_file(dirpath + '/' + f)
                    face_landmarks = face_recognition.face_landmarks(image)[0]
                    eye_left = face_landmarks['left_eye']
                    eye_left = ((eye_left[0][0] + eye_left[1][0]) // 2, (eye_left[0][1] + eye_left[1][1]) // 2)
                    eye_right = face_landmarks['right_eye']
                    eye_right = ((eye_right[0][0] + eye_right[1][0]) // 2, (eye_right[0][1] + eye_right[1][1]) // 2)
                    eye_landmarks[f[4:7]] = (eye_left, eye_right)
                    cropXY_size = ReturnCropXYSize(Image.fromarray(image), eye_left, eye_right, offset_pct=(0.35, 0.35), dest_sz=(128, 128))
                    crop_xy[f[4:7]] = cropXY_size[0]
                    crop_size[f[4:7]] = cropXY_size[1]
                except Exception as e:
                    print(e)
                    print(f[4:7], ' >>> error ')
                    PrintException()
    for dirpath, dirnames, filenames in os.walk(dir):
        print('dirpath: ' + dirpath)
        filestoCheck = []
        for f in filenames:
            if 'P00A' in f:
                filestoCheck.append(f);
        for f in filestoCheck:
            if angle_name in f:
                try:
                    print(f)
                    # landmark 찾기
                    image = face_recognition.load_image_file(dirpath + '/' + f)
                    eye_left = eye_landmarks[f[4:7]][0]
                    eye_right = eye_landmarks[f[4:7]][1]

                    # save
                    save_path = os.path.join(save_dir, f)
                    print("save_path:", save_path)
                    pil_image = Image.fromarray(image)
                    CropFace(pil_image, eye_left=eye_left, eye_right=eye_right, offset_pct=(0.35, 0.35),
                             dest_sz=(128, 128), in_crop_xy = crop_xy[f[4:7]], in_crop_size = crop_size[f[4:7]]) \
                        .save(save_path)
                except Exception as e:
                    print(e)
                    print(f[4:7], ' >>> error ')
                    PrintException()

# 두 개의 이미지 받아서 이어 주기
def concatAndSave(img1_dir, img2_dir):
    for dirpath, dirnames, filenames in os.walk(img2_dir):
        for f in filenames:
            print(f)
            try:
                img1 = cv2.imread(img1_dir + f, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img2_dir + f, cv2.IMREAD_GRAYSCALE)
                addh = cv2.hconcat([img1, img2])

                cv2.imwrite(f, addh)
            except Exception as e:
                print(e)
                print(img1_dir+f, img2_dir+f)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#                   디렉터리 이미지 삭제 필터링 이동                  #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 아무것도 없는 디렉터리 삭제
def remove_zero_size_dir(start_path):
    for dirpath, dirnames, filenames in os.walk(start_path):
        size = 0
        for f in filenames:
            fp = os.path.join(dirpath, f)
            size += os.path.getsize(fp)
        if size == 0 and len(dirnames) == 0:
            print(dirpath, dirnames, filenames)
            shutil.rmtree(dirpath)

# 사이즈가 없는 파일 삭제
def remove_zero_size_file(start_path):
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if len(dirnames) == 0 and os.path.getsize(fp) == 0:
                print(fp)
                os.remove(fp)

# 이미지 이동
def move_images(start_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for dirpath, dirnames, filenames in os.walk(start_path):
        if len(dirnames) == 0:
            print(dirpath)
            for f in filenames:
                fp = os.path.join(dirpath, f)
                path = fp.replace('\\', '/').split('/')
                type = path[-3]
                session = path[-4]
                num = path[-2]
                dest_dir = os.path.join(save_dir, type)
                if not os.path.exists(dest_dir):
                    os.mkdir(dest_dir)
                dest = dest_dir + '/' + session + '_' + num + '_' + f
                shutil.copy(fp, dest)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#                    이미지 처리 (이미지 절반 픽셀)                  #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 이미지 절반 픽셀에 검정색 뿌리기
def halfBlack(img_dir):
    img = cv2.imread(img_dir)
    for i in range(128):
        for j in range(64):
            img[i, 64+j] = [0,0,0]
    return img

def halfCopy(img_dir):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    for i in range(128):
        for j in range(64):
            img[i, 64 + j] = img[i, 64 - j]
    return img

def halfRandom(img_dir):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    for i in range(128):
        for j in range(64):
            img[i, 64 +j] = random.randint(0, 256)
    return img

def blurCenter(img):
    for i in range(128):
        img[i, 63] = statistics.mean([img[i, 60], img[i, 61], img[i, 62], img[i, 63]])
        img[i, 64] = statistics.mean([img[i, 61], img[i, 62], img[i, 63], img[i, 64]])
        img[i, 65] = img[i, 64]
        img[i, 66] = img[i, 63]
    for i in range (126):
        img[i+1, 62] = statistics.mean([img[i, 61], img[i, 62], img[i, 63],
                                        img[i+1, 61], img[i+1, 62], img[i+1, 63],
                                        img[i+2, 61], img[i+2, 62], img[i+2, 63]])
        img[i+1, 63] = statistics.mean([img[i, 62], img[i, 63], img[i, 64],
                                          img[i + 1, 62], img[i + 1, 63], img[i + 1, 64],
                                          img[i + 2, 62], img[i + 2, 63], img[i + 2, 64]])
        img[i+1, 64] = statistics.mean([img[i, 63], img[i, 64], img[i, 65],
                                          img[i + 1, 63], img[i + 1, 64], img[i + 1, 65],
                                          img[i + 2, 63], img[i + 2, 64], img[i + 2, 65]])
    return img

def pushRightHalfCopy(img_dir, pix):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    returnIMG = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    for i in range(128):
        for j in range(65):
            if j < pix:
                # do nothing
                img[i, j] = 0
            else:
                returnIMG[i, j] = img[i, j-pix]
    for i in range(128):
        for j in range(64):
            returnIMG[i, 64 + j] = returnIMG[i, 64 - j]
    return returnIMG

def maskAndSave(img_dir):
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for f in filenames:
            image_path = os.path.join(dirpath, f)
            print(image_path)
            try:
                img = halfCopy(image_path)
                img = blurCenter(img)
                cv2.imwrite(f[:-3]+'png', img)
            except Exception as e:
                print(e)
                print(image_path, 'ignored')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#                   트레인 및 테스트 데이터셋 나누기                  #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def random_select_move(start_path, train, test):
    subjects = []
    sub_dict = {}
    subjects_train = []
    subjects_test = []

    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            subject_name = f[0:8]
            if subject_name not in subjects:
                subjects.append(f[0:8])
                sub_dict[subject_name] = [f[9:12]]
            else:
                sub_dict[subject_name].append(f[9:12])

    num_subject = len(subjects)
    train_num = int(num_subject * train)
    test_num = num_subject - train_num

    random.shuffle(subjects)
    subjects_train = subjects[:train_num-1]
    subjects_test = subjects[train_num:]

    print("total subjects:", len(subjects))
    print("train subjects:", len(subjects_train))
    print("test subjects:", len(subjects_test))

    for img in subjects_train:
        for num in sub_dict[img]:
            num_of_img = '_' + num
            filename = img + num_of_img + '.png'
            try:
                shutil.move('./' + filename, '../aligned_train/' + filename)
            except Exception as e:
                print(e)

    for img in subjects_test:
        for num in sub_dict[img]:
            num_of_img = '_' + num
            filename = img + num_of_img + '.png'
            try:
                shutil.move('./' + filename, '../aligned_test/'+ filename)
            except Exception as e:
                print(e)