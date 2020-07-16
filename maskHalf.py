import numpy as np
import cv2, os
import statistics
import random
from skimage.restoration import denoise_nl_means, estimate_sigma
import maskHalf_with_landmark

def justLoad(img_dir):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    return img

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

def halfAlphaMask(img_dir, a_rate):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    for i in range(128):
        for j in range(64):
            if (img[i, 64 + j] - 255 * a_rate > 0):
                img[i, 64 + j] = img[i, 64 + j] - (255*a_rate)
            else:
                img[i, 64 + j] = 0;
    return img

# 가운데 그림자 완화하려고...
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
        # img[i, 65] = img[i, 64]
        # img[i, 66] = img[i, 63]
        # img[i, 67] = img[i, 62]

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

def NLfilter(img):
    float_img = img.astype(np.float32)
    sigma_est = np.mean(estimate_sigma(float_img))
    denoise_img = denoise_nl_means(float_img, h=3*sigma_est,
                                    patch_size=5,
                                    patch_distance=3)
    # cv2.imshow('denoised', denoise_img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return denoise_img

def maskAndSave(img_dir, save_dir_name):
    if not os.path.isdir('../'+save_dir_name):
        os.mkdir('../'+save_dir_name)
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for f in filenames:
            image_path = os.path.join(dirpath, f)
            print(image_path)
            try:
                # img = justLoad(image_path)
                # img = halfBlack(image_path)
                # img = halfCopy(image_path)
                img = halfAlphaMask(image_path, 0.2)
                # img = halfAlphaMaskGradient(image_path, 0.3)
                img = NLfilter(img)
                # img = pushRightHalfCopy(image_path, 4)
                # img = halfRandom(image_path)
                # img = maskHalf_with_landmark.maskHalf_landmark(image_path)
                # img = blurCenter(img.astype(int))
                cv2.imwrite('../' + save_dir_name + '/'+f[:-3]+'png', img)
            except Exception as e:
                print(e)
                print(image_path, 'ignored')

def testAndShow(img_dir):
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for f in filenames:
            image_path = os.path.join(dirpath, f)
            print(image_path)
            try:
                img = halfAlphaMask(image_path, 0.2)
                img = NLfilter(img)
                # img = blurCenter(img)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                print(f)
            except Exception as e:
                print(e)
                print(image_path, 'ignored')

maskAndSave('./', 'result_train')