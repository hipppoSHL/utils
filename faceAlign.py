# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Philipp Wagner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the author(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# https://bytefish.de/blog/aligning_face_images/

import os, sys, math, numpy
from PIL import Image


def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


def CropFace(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    print('eye_left:', eye_left, 'eye_right:', eye_right)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


def ModifiedCropFace(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    
    npimage = numpy.array(image)
    face_landmarks = face_recognition.face_landmarks(npimage, model='small')[0]
    eye_left = face_landmarks['left_eye']
    eye_left = ((eye_left[0][0] + eye_left[1][0]) // 2, (eye_left[0][1] + eye_left[1][1]) // 2)

    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


def ModifiedCropFace2(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70), in_crop_xy=(0,0), in_crop_size=(0,0)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)

    # npimage = numpy.array(image)
    # face_landmarks = face_recognition.face_landmarks(npimage, model='small')[0]
    # eye_left = face_landmarks['left_eye']
    # eye_left = ((eye_left[0][0] + eye_left[1][0]) // 2, (eye_left[0][1] + eye_left[1][1]) // 2)

    # crop the rotated image
    crop_xy = in_crop_xy
    crop_size = in_crop_size
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image

def ReturnCropXYSize(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)

    npimage = numpy.array(image)
    face_landmarks = face_recognition.face_landmarks(npimage, model='small')[0]
    eye_left = face_landmarks['left_eye']
    eye_left = ((eye_left[0][0] + eye_left[1][0]) // 2, (eye_left[0][1] + eye_left[1][1]) // 2)

    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    return crop_xy, crop_size


def align_face_with_text_file(txt_path, save_dir):
    dir_path = os.path.split(txt_path)[0]
    f = open(txt_path, 'r')
    lines = f.readlines()
    for line in lines:
        parse = line.split()
        image_path = os.path.join(dir_path, parse[0])
        if not os.path.exists(image_path):
            print(image_path, 'not exist')
            continue
        eye_left = tuple(map(int, parse[1:3]))
        eye_right = tuple(map(int, parse[3:5]))
        try:
            image = Image.open(image_path)
            paths = image_path.split('\\')
            save_name = paths[-4] + '_' + paths[-2] + '_' + paths[-1]
            if not os.path.exists(os.path.join(save_dir, paths[-3])):
                os.makedirs(os.path.join(save_dir, paths[-3]))
            save_path = os.path.join(save_dir, paths[-3], save_name)
            CropFace(image, eye_left=eye_left, eye_right=eye_right, offset_pct=(0.25, 0.25), dest_sz=(128, 128)) \
                .save(save_path)
        except:
            print(image_path, 'ignored')
    f.close()


import cv2, face_recognition


def align_face_with_face_landmarks(image_dir, save_dir):
    for dirpath, dirnames, filenames in os.walk(image_dir):
        if len(dirnames) == 0:
            for f in filenames:
                image_path = os.path.join(dirpath, f)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_landmarks = face_recognition.face_landmarks(image, model='small')[0]
                    eye_left = face_landmarks['left_eye']
                    eye_left = ((eye_left[0][0] + eye_left[1][0]) // 2, (eye_left[0][1] + eye_left[1][1]) // 2)
                    eye_right = face_landmarks['right_eye']
                    eye_right = ((eye_right[0][0] + eye_right[1][0]) // 2, (eye_right[0][1] + eye_right[1][1]) // 2)

                    paths = image_path.split('\\')
                    save_name = paths[-4] + '_' + paths[-2] + '_' + paths[-1]
                    if not os.path.exists(os.path.join(save_dir, paths[-3])):
                        os.makedirs(os.path.join(save_dir, paths[-3]))
                    save_path = os.path.join(save_dir, paths[-3], save_name)
                    pil_image = Image.fromarray(image)
                    CropFace(pil_image, eye_left=eye_left, eye_right=eye_right, offset_pct=(0.25, 0.25),
                             dest_sz=(128, 128)) \
                        .save(save_path)
                except Exception as e:
                    print(e)
                    print(image_path, 'ignored')

def alignAndSave(image_dir, save_dir):
    for dirpath, dirnames, filenames in os.walk(image_dir):
        for f in filenames:
            image_path = os.path.join(dirpath, f)
            paths = image_path.split('//')
            try:
                # mask right half side with mirror 내가 넣은 코드
                aligned_face = face_recognition.load_image_file(image_path)
                aligned_face_landmarks = face_recognition.face_landmarks(aligned_face)
                print(aligned_face_landmarks)
                aligned_left = aligned_face_landmarks[0]['left_eye']
                aligned_left = aligned_left[4][0]
                aligned_right = aligned_face_landmarks[0]['right_eye']
                aligned_right = aligned_right[4][0]
                aligned_center = (aligned_left + aligned_right) / 2
                mirrored_face = aligned_face[0]
                print(mirrored_face)
                for i in range(128):
                    for j in range(int(128-aligned_center)):
                        mirrored_face[i][aligned_center + j] = aligned_face[i][aligned_center - j]
                mirrored_face.save(f)

            except Exception as e:
                print(e)
                print(image_path, 'ignored', ">>>>>")
                _, _, tb = sys.exc_info()  # tb -> traceback object print 'file name = ', __file__ print 'error line No = {}'.format(tb.tb_lineno) print e
                print('error line No = {}'.format(tb.tb_lineno))

# alignAndSave('.','.')
