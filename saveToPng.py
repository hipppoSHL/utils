from PIL import Image
import os

def savePNG(img_dir):
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for f in filenames:
            # 이미지 열기
            im = Image.open(f)
            # 이미지 PNG로 저장
            im.save(f[:-3] + 'png')

savePNG('./')

