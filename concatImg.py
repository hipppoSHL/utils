import cv2, os
import numpy as np

# 두 개의 이미지 받아서 이어 주기
def concatAndSave(img1_dir, img2_dir, save_dir_name):
    for dirpath, dirnames, filenames in os.walk(img2_dir):
        for f in filenames:
            print(f)
            try:
                img1 = cv2.imread(img1_dir + f, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img2_dir + f, cv2.IMREAD_GRAYSCALE)
                addh = cv2.hconcat([img1, img2])

                cv2.imwrite('./'+save_dir_name+'/'+f, addh)
            except Exception as e:
                print(e)
                print(img1_dir+f, img2_dir+f)

concatAndSave('./stretched_alpha03_blended_gradient/test/', './contrast_stretched/test/', './concat_stretched_alpha03_blended_gradient/test/')
# concatAndSave('../original/', '../only_mirrored/')
# concatAndSave('/home/soohyeonlee/project/FaceDataset/ExtendedYaleB_modified/실험사용할것만모아놓음/original/', '/home/soohyeonlee/project/FaceDataset/ExtendedYaleB_modified/실험사용할것만모아놓음/yale_b_half_random/')
# concatAndSave('./original/', './yale_b_centerNL/', './yale_b_centerNL_concat/')