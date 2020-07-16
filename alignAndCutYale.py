from faceAlign import ModifiedCropFace2 as CropFace
from faceAlign import ReturnCropXYSize
from PrintException import PrintException
from PIL import Image

import os, cv2, face_recognition
from PIL import Image

# only applies to yale extended b database
def temp(angle_name, dir, save_dir):
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
                    # save
                    # save_path = os.path.join(save_dir, f)
                    # print(save_path)
                    # pil_image = Image.fromarray(image)
                    # CropFace(pil_image, eye_left=eye_left, eye_right=eye_right, offset_pct=(0.35, 0.35),
                    #          dest_sz=(128, 128)) \
                    #     .save(save_path)
                except Exception as e:
                    print(e)
                    print(f[4:7], ' >>> 랜드마크 찾을 수 없거나 something wrong... ')
                    PrintException()
            # elif 'P00A+050E+00' in f:
            #     try:
            #         print(f)
            #         # landmark 찾기
            #         image = face_recognition.load_image_file(dirpath + '/' + f)
            #         eye_left = eye_landmarks[f[4:7]][0]
            #         eye_right = eye_landmarks[f[4:7]][1]

            #         # save
            #         save_path = os.path.join(save_dir, f)
            #         print(save_path)
            #         pil_image = Image.fromarray(image)
            #         CropFace(pil_image, eye_left=eye_left, eye_right=eye_right, offset_pct=(0.35, 0.35),
            #                  dest_sz=(128, 128)) \
            #             .save(save_path)
            #     except Exception as e:
            #         print(e)
            #         print(f[4:7], ' >>> 랜드마크 찾을 수 없거나 something wrong... ')
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
                    print(f[4:7], ' >>> 랜드마크 찾을 수 없거나 something wrong... ')
                    PrintException()

# angle_name='P00A+035E+15'
# temp('P00A+060E-20', './', '/home/soohyeonlee/project/FaceDataset/ExtendedYaleB_modified/Darker/P00A+060E-20/')
# temp('P00A+070E+00', './', '/home/soohyeonlee/project/FaceDataset/ExtendedYaleB_modified/Darker/P00A+070E+00/')
# temp('P00A+085E+20', './', '/home/soohyeonlee/project/FaceDataset/ExtendedYaleB_modified/Darker/P00A+085E+20/')
# temp('P00A+110E+15', './', '/home/soohyeonlee/project/FaceDataset/ExtendedYaleB_modified/Darker/P00A+110E+15/')
temp('P00A-060E-20', './', '../bb/')
temp('P00A-070E-45', './', '../bb/')
temp('P00A+000E+90', './', '../bb/')