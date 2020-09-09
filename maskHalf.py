import numpy as np
import cv2, os
import statistics
import random
from skimage.restoration import denoise_nl_means, estimate_sigma

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

def halfAlphaMask2(img1_dir, img2_dir, a_rate):
    # img2 absolute path: /home/soohyeonlee/lab-workspace/Utils/tmp/tmp.png
    img1 = cv2.imread(img1_dir, cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(img2_dir, cv2.IMREAD_GRAYSCALE)
    dest = img1
    for i in range (128):
        for j in range(64):
            dest[i, 64 + j] = img1[i, 64 + j] * a_rate + 0 * (1-a_rate)
    return dest

def halfAlphaMask3(img, a_rate):
    dest = img
    for i in range (128):
        for j in range(64):
            dest[i, 64 + j] = img[i, 64 + j] * a_rate + 0 * (1-a_rate)
    return dest

def halfAlphaMask_Gradient(img_dir, a_rate):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    for i in range(128):
        for j in range(64):
            if (img[i, 64 + j] - (255 * a_rate)*j/20 > 0):
                img[i, 64 + j] = img[i, 64 + j] - (255*a_rate)*j/20
            else:
                img[i, 64 + j] = 0
    return img

def halfAlphaMask_Gradient2(img1_dir, img2_dir, a_rate):
    img1 = cv2.imread(img1_dir, cv2.IMREAD_GRAYSCALE)
    # img2 absolute path: /home/soohyeonlee/lab-workspace/Utils/tmp/tmp.png
    # img2 = cv2.imread(img2_dir, cv2.IMREAD_GRAYSCALE)
    dest = img1
    for i in range(128):
        for j in range(64):
            cur_rate = 0 + a_rate * (j / 64)
            dest[i, 64 + j] = img1[i, 64 + j] * (1 - cur_rate) + 0 * (cur_rate)
    return dest


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
    return denoise_img

def NLfilterCenter(img):
    float_img = img.astype(np.float32)
    sigma_est = np.mean(estimate_sigma(float_img))
    denoise_img = denoise_nl_means(float_img, h=3 * sigma_est,
                                   patch_size=5,
                                   patch_distance=3)
    for i in range(128):
        float_img[i, 62] = denoise_img[i, 62]
        float_img[i, 63] = denoise_img[i, 63]
        float_img[i, 64] = denoise_img[i, 64]
        float_img[i, 65] = denoise_img[i, 65]
        # float_img[i, 62] = 0
        # float_img[i, 63] = 0
        # float_img[i, 64] = 0
        # float_img[i, 65] = 0

    return float_img

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
                # img = NLfilterCenter(img)
                # img = halfAlphaMask(image_path, 0.2)
                # img = halfAlphaMask2(image_path, '/home/soohyeonlee/lab-workspace/Utils/tmp/tmp.png', 0.1)
                # img = halfAlphaMask_Gradient(image_path, 0.25)
                img = halfAlphaMask_Gradient2(image_path, '/home/soohyeonlee/lab-workspace/Utils/tmp/tmp.png', 0.7)
                # img = NLfilter(img)
                # img = pushRightHalfCopy(image_path, 4)
                # img = halfRandom(image_path)
                # img = maskHalf_with_landmark.maskHalf_landmark(image_path)
                # img = blurCenter(img.astype(int))

                # img1 = halfCopy(image_path)
                # img2 = justLoad(image_path)
                # # blend image 1 & 2
                # img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
                # img = halfAlphaMask3(img, 0.5)

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
                img = halfAlphaMask_Gradient(image_path, 0.2)
                # img = NLfilter(img)
                # img = blurCenter(img)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(f)
            except Exception as e:
                print(e)
                print(image_path, 'ignored')

# maskAndSave('./', 'aligned_nl_alpha03')
maskAndSave('./', 'stretched_alpha03_blended_gradient_test')