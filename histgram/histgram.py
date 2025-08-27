import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_histogram(hist_b, hist_g, hist_r, filename):
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.plot(hist_b, color='b', label='Blue')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_r, color='r', label='Red')
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



img_a = cv2.imread("./img/set2_a.png")
img_b = cv2.imread("./img/set2_b.png")
print(img_a.shape)
print(img_b.shape)

# 画像サイズの統一
height = img_b.shape[0]
width = img_b.shape[1]
img_a = cv2.resize(img_a , (int(width), int(height)))

# ヒストグラム計算
hist_a_b = cv2.calcHist([img_a], [0], None, [256], [0, 256])
hist_b_b = cv2.calcHist([img_b], [0], None, [256], [0, 256])
hist_a_g = cv2.calcHist([img_a], [1], None, [256], [0, 256])
hist_b_g = cv2.calcHist([img_b], [1], None, [256], [0, 256])
hist_a_r = cv2.calcHist([img_a], [2], None, [256], [0, 256])
hist_b_r = cv2.calcHist([img_b], [2], None, [256], [0, 256])

save_histogram(hist_a_b, hist_a_g, hist_a_r, "./set2_a_hist.png")
save_histogram(hist_b_b, hist_b_g, hist_b_r, "./set2_b_hist.png")
