import numpy as np
import cv2
import random
import os
# calculate means and std
from tqdm import tqdm
import numpy as np

train_path = 'D:/workspace/crnet/data/coco/images/train/'
val_path = 'D:/workspace/crnet/data/coco/images/val/'
img_name = os.listdir(train_path)
CNum = len(img_name)  # select images 取前10000张图片作为计算样本

img_h, img_w = 511, 511
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []


random.shuffle(img_name)  # shuffle images

for i in tqdm(range(CNum)):
    file_name = img_name[i]
    img_path = os.path.join(train_path, file_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_h, img_w))  # 将图片进行裁剪[32,32]
    img = img[:, :, :, np.newaxis]
    imgs = np.concatenate((imgs, img), axis=3)


imgs = imgs.astype(np.float32) / 255.

for i in tqdm(range(3)):
    pixels = imgs[:, :, i, :].ravel()  # flatten
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 : BGR
means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))


# todo 特征值和特征向量
imgs = np.zeros([img_w, img_h, 3, 1])

for i in tqdm(range(CNum)):
    file_name = img_name[i]
    img_path = os.path.join(train_path, file_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_h, img_w))
    img = img[:, :, :, np.newaxis]
    imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32) / 255.

pixels = imgs[:, :, 2, :].ravel()  # flatten
scaled_R = pixels - means[0]  # R
pixels = imgs[:, :, 1, :].ravel()  # flatten
scaled_G = pixels - means[1]  # G
pixels = imgs[:, :, 0, :].ravel()  # flatten
scaled_B = pixels - means[2]  # B

cov = np.cov((scaled_R, scaled_G), scaled_B)  # 求三个变量的协方差
eig_val, eig_vec = np.linalg.eig(cov)
print(eig_val)
print(eig_vec)