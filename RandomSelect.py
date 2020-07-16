import os
import random
import shutil

def random_select_move(start_path, train, test):
    subjects = []
    sub_dict = {}
    subjects_train = []
    subjects_test = []

    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            subject_name = f[0:8]
            if subject_name not in subjects:
                subjects.append(f[0:8])
                sub_dict[subject_name] = [f[9:12]]
            else:
                sub_dict[subject_name].append(f[9:12])

    num_subject = len(subjects)
    train_num = int(num_subject * train)
    test_num = num_subject - train_num

    random.shuffle(subjects)
    subjects_train = subjects[:train_num-1]
    subjects_test = subjects[train_num:]

    print("total subjects:", len(subjects))
    print("train subjects:", len(subjects_train))
    print("test subjects:", len(subjects_test))

    for img in subjects_train:
        for num in sub_dict[img]:
            num_of_img = '_' + num
            filename = img + num_of_img + '.png'
            try:
                shutil.move('./' + filename, '../aligned_train/' + filename)
            except Exception as e:
                print(e)

    for img in subjects_test:
        for num in sub_dict[img]:
            num_of_img = '_' + num
            filename = img + num_of_img + '.png'
            try:
                shutil.move('./' + filename, '../aligned_test/'+ filename)
            except Exception as e:
                print(e)


random_select_move('./', 0.8, 0.2)
