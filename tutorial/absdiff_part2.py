"""
https://qiita.com/jun_higuche/items/e3c3263ba50ea296f7bf
画像比較して異なる箇所を別画像で表示
↓を参考に
https://note.nkmk.me/python-opencv-numpy-image-difference/
"""

import cv2, os
import numpy as np


img_1 = cv2.imread("../img/set2_a.png")
img_2 = cv2.imread("../img/set2_b.png")

height = img_1.shape[0]
width = img_1.shape[1]

img_size = (int(width), int(height))

# 画像をリサイズする
image1 = cv2.resize(img_1, img_size)
image2 = cv2.resize(img_2, img_size)

# ２画像の差異を計算
im_diff = image1.astype(int) - image2.astype(int)

# 単純に差異をそのまま出力する
cv2.imwrite("01_diff.png", im_diff)

# 差異が無い箇所を中心（灰色：128）とし、そこからの差異を示す
cv2.imwrite("02_diff_center.png", im_diff + 128)

# 差異が無い箇所を中心（灰色：128）とし、差異を2で割った商にする（差異を-128～128にしておきたいため）
im_diff_center = np.floor_divide(im_diff, 2) + 128
cv2.imwrite("03_diff_center.png", im_diff_center)
