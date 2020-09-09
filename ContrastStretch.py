import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def contrast_stretch(img_dir):
    gray_img = Image.open(img_dir).convert("LA")
    row = gray_img.size[0]
    col = gray_img.size[1]
    stretch_img = Image.new("L", (row, col))  # 새 흑백이미지를 생성.
    high = 0
    low = 255

    for x in range(0, row):
        for y in range(0, col):
            if high < gray_img.getpixel((x, y))[0]:
                high = gray_img.getpixel((x, y))[0]
            if low > gray_img.getpixel((x, y))[0]:
                low = gray_img.getpixel((x, y))[0]
    for x in range(0, row):
        for y in range(0, col):
            stretch_img.putpixel((x, y), int((gray_img.getpixel((x, y))[0] - low) * 255 / (high - low)))

    # y = gray_img.histogram()
    # y = y[0:256]
    # x = np.arange(len(y))
    # plt.title("original hist")
    # plt.bar(x, y)
    # plt.show()

    # y = stretch_img.histogram()
    # x = np.arange(len(y))
    # plt.title("stretch hist")
    # plt.bar(x, y)
    # plt.show()
    return stretch_img

def convertAndSave(img_dir, save_dir_name):
    if not os.path.isdir('../' + save_dir_name):
        os.mkdir('../' + save_dir_name)
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for f in filenames:
            image_path = os.path.join(dirpath, f)
            print(image_path)
            try:
                img = contrast_stretch(image_path)
                img.save('../' + save_dir_name + '/'+f[:-3]+'png')
            except Exception as e:
                print(e)
                print(image_path, 'ignored')

convertAndSave('./', 'stretched_test')