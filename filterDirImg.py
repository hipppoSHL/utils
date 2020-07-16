import os, shutil, random, traceback, glob
import cv2, face_recognition
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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

move_images("../FaceDataset/Casia/NIR-VIS-2.0/s1/VIS/", "../ModifiedFile")
move_images("../FaceDataset/Casia/NIR-VIS-2.0/s2/VIS/", "../ModifiedFile")
move_images("../FaceDataset/Casia/NIR-VIS-2.0/s3/VIS/", "../ModifiedFile")
move_images("../FaceDataset/Casia/NIR-VIS-2.0/s4/VIS/", "../ModifiedFile")
remove_zero_size_file("../ModifiedFile")
