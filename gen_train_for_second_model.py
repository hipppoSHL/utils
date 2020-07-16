import cv2, os
import numpy as np

# only applies to yale extended b database
def temp(dir):
    for dirpath, dirnames, filenames in os.walk(dir):
        filestoCheck = []
        cnt = 0
        for f in filenames:
            if 'real_B' in f:
                filestoCheck.append(f[:-11])
                cnt += 1
        for f in filestoCheck:
            # print(f)
            try:
                real_B = cv2.imread(dir + f + '_real_B.png', cv2.IMREAD_GRAYSCALE)
                fake_B = cv2.imread(dir + f + '_fake_B.png', cv2.IMREAD_GRAYSCALE)
                addh = cv2.hconcat([real_B, fake_B])
                cv2.imwrite('./concat/' + f +'.png', addh)

            except Exception as e:
                print(e)
                print(f, 'ignored')


        print('total concatenated:', cnt, 'items')
        print('----------------------------------------------------')

temp('./')