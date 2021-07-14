import os

directory = os.getcwd()
os.chdir(directory)
train_test_ratio = 0.15
path_img = []
path_txt = []
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.jpg'):
            path_to_save = directory + '/' + f
            path_img.append(path_to_save)
        if f.endswith('.txt'):
            path_to_save = directory + '/' + f
            path_txt.append(path_to_save)

path_img.sort()
path_txt.sort()
path_test_img = path_img[:int(len(path_img) * train_test_ratio)]
path_train_img = path_img[int(len(path_img) * train_test_ratio):]
path_test_txt = path_txt[:int(len(path_img) * train_test_ratio)]
path_train_txt = path_txt[int(len(path_img) * train_test_ratio):]
print(path_train_img)
print(path_train_txt)
print(path_test_img)
print(path_test_txt)
print("Training data:")
with open('train.txt', 'w') as train_txt:
    for i in range(len(path_train_img)):
        train_txt.write('{} '.format(path_train_img[i]))
        print(path_train_img[i])
        with open(path_train_txt[i]) as txt:
            holder = txt.readlines()
            for j in range(len(holder)):
                coords = holder[j].split()
                print(coords)
                x1 = float(coords[1])
                y1 = float(coords[2])
                x2 = float(coords[3])
                y2 = float(coords[4])
                class_type = int(coords[0])
                train_txt.write('{},{},{},{},{} '.format(x1, y1, x2, y2, class_type))
        train_txt.write('\n')
print(" Test data:")
with open('test.txt', 'w') as test_txt:
    for i in range(len(path_test_img)):
        test_txt.write('{} '.format(path_test_img[i]))
        with open(path_test_txt[i]) as txt:
            holder = txt.readlines()
            for j in range(len(holder)):
                coords = holder[j].split()
                print(coords)
                x1 = float(coords[1])
                y1 = float(coords[2])
                x2 = float(coords[3])
                y2 = float(coords[4])
                class_type = int(coords[0])
                test_txt.write('{},{},{},{},{} '.format(x1, y1, x2, y2, class_type))
        test_txt.write('\n')

