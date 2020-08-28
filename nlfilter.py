import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.image as mpimg


def bfunc(i, j, fw, fh, image, sigma1, sigma2, bilateralWFilter):
    imgwork = image[i - fh // 2:i + 1 + fh // 2, j - fw // 2:j + 1 + fw // 2, :]

    bilateralIFilter = ((imgwork - image[i, j, :]) ** 2) / (2 * (sigma1 ** 2))

    bilateralFilter = np.exp(-1 * bilateralIFilter) * bilateralWFilter
    bilateralFilter = bilateralFilter / np.sum(bilateralFilter, axis=(0, 1))
    return np.sum(np.multiply(imgwork, bilateralFilter), axis=(0, 1))


def bilateralFilterConv(image, fw, fh):
    size = image.shape
    sigma1 = 40
    sigma2 = 40
    bilateral1 = 2 * 3.14 * sigma2 * sigma2 * gaussFilter((fw, fh), sigma2)
    if len(image.shape) < 3 or image.shape[2] == 1:
        bilateralWFilter = np.resize(bilateral1, (*bilateral1.shape, 1))
    else:
        bilateralWFilter = np.stack([bilateral1, bilateral1, bilateral1], axis=2)

    out = np.zeros((size[0] - 2 * fw + 1, size[1] - 2 * fh + 1, size[2]))
    for i in range(size[0] - 2 * fh + 1):
        for j in range(size[1] - 2 * fw + 1):
            out[i, j, :] = bfunc(i + fw - 1, j + fh - 1, fw, fh, image, sigma1, sigma2, bilateralWFilter)

    if id == 1:
        return np.resize(out, (out.shape[0], out.shape[1])).astype(np.uint8)
    else:
        return out.astype(np.uint8)


# gaussian filter
def gfunc(x,y,sigma):
    return (math.exp(-(x**2 + y**2)/(2*(sigma**2))))/(2*3.14*(sigma**2))

def gaussFilter(size, sigma):
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = gfunc(i-size[0]//2,j-size[1]//2, sigma )
    return out/np.sum(out)


# Non Local Mean filter

def nlmfunc(i, j, fw, fh, nw, nh, image, sigma1, sigma2, nlmWFilter):
    imgmain = image[i - fh // 2:i + 1 + fh // 2, j - fw // 2:j + 1 + fw // 2, :]

    nlmFilter = 0
    for p in range(-(nh // 2), 1 + (nh // 2)):
        for q in range(-(nw // 2), 1 + (nw // 2)):
            imgneighbour = image[i + p - fh // 2: i + 1 + p + fh // 2, j + q - fw // 2:j + 1 + q + fw // 2, :]
            nlmIFilter = ((imgmain - imgneighbour) ** 2) / (2 * (sigma1 ** 2))
            nlmFilter += np.exp(-1 * nlmIFilter)

    nlmFilter = nlmFilter / np.sum(nlmFilter, axis=(0, 1))
    nlmFilter = nlmFilter * nlmWFilter
    nlmFilter = nlmFilter / np.sum(nlmFilter, axis=(0, 1))
    return np.sum(np.multiply(imgmain, nlmFilter), axis=(0, 1))


def nlmFilterConv(image, fw, fh, nw, nh):
    size = image.shape
    sigma1 = 20
    sigma2 = 20
    nlmWFilter1 = 2 * 3.14 * sigma2 * sigma2 * gaussFilter((fw, fh), sigma2)
    if len(image.shape) < 3 or image.shape[2] == 1:
        nlmWFilter = np.resize(nlmWFilter1, (*nlmWFilter1.shape, 1))
    else:
        nlmWFilter = np.stack([nlmWFilter1, nlmWFilter1, nlmWFilter1], axis=2)

    out = np.zeros((size[0] - 2 * fw + 1 - nw // 2, size[1] - 2 * fh + 1 - nh // 2, size[2]))
    for i in range(nh // 2, size[0] - 2 * fh + 1 - nh // 2):
        for j in range(nw // 2, size[1] - 2 * fw + 1 - nw // 2):
            out[i, j, :] = nlmfunc(i + fw - 1, j + fh - 1, fw, fh, nw, nh, image, sigma1, sigma2, nlmWFilter)

    out[0:nh // 2, :, :] = out[nh // 2, :, :]
    out[:, 0:nw // 2, :] = out[:, nw // 2, :, np.newaxis]
    if id == 1:
        return np.resize(out, (out.shape[0], out.shape[1])).astype(np.uint8)
    else:
        return out.astype(np.uint8)