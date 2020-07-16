import os
from PIL import Image, ImageDraw
import face_recognition

def draw_landmark(img_dir, save_dir):
    total = 0
    success = 0
    fail = 0
    flag = 0
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for f in filenames:
            image_path = os.path.join(dirpath, f)
            total += 1
            try:
                image = face_recognition.load_image_file(image_path)
                face_landmarks_list = face_recognition.face_landmarks(image)
                pil_image = Image.fromarray(image)
                for face_landmarks in face_landmarks_list:
                    for face_landmark in face_landmarks:
                        if flag == 0:
                            success += 1
                            flag = 1
                        d = ImageDraw.Draw(pil_image, 'RGB')
                        d.line(face_landmarks[face_landmark], fill='white', width=3)
                if flag == 0:
                    fail += 1
                pil_image.save(save_dir+f)
                flag = 0
                # pil_image.save('C:\\Users\\405B\\Desktop\\s2_00005_001_landmarked.png')
            except Exception as e:
                print(e)
                print(f, 'ignored!')
    print('success / total:', success, '/', total)
    print('fail:', fail)

    f = open(save_dir+'info.txt', 'w')
    data = f"{success}/{total} (success/total)\nfailed: {fail}"
    f.write(data)
    f.close()

draw_landmark('/home/soohyeonlee/project/casia_example_vis/', '/home/soohyeonlee/project/casia_example_vis_landmark/')

