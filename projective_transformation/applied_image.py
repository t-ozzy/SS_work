import cv2
import numpy as np

img_1 = cv2.imread("./img/set2_b.png")
gamma = 1.8  # この値で明るさが変わる
gamma_cvt = np.zeros((256, 1), dtype="uint8")
for i in range(256):
    gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
img_1 = cv2.LUT(img_1, gamma_cvt)
img_2 = cv2.imread("./aligned_image.png")
if img_1.shape != img_2.shape:
    exit()

# グレースケール変換
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
cv2.imwrite("tmp_1_1.png", gray_1)
cv2.imwrite("tmp_1_2.png", gray_2)

# ノイズ除去
gray_1 = cv2.GaussianBlur(gray_1, (5, 5), 0)
gray_2 = cv2.GaussianBlur(gray_2, (5, 5), 0)
cv2.imwrite("tmp_2_1.png", gray_1)
cv2.imwrite("tmp_2_2.png", gray_2)

# コントラスト限定適応ヒストグラム平坦化
clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
gray_1 = clahe.apply(gray_1)
gray_2 = clahe.apply(gray_2)
cv2.imwrite("tmp_3_1.png", gray_1)
cv2.imwrite("tmp_3_2.png", gray_2)

# 画像を引き算
img_diff = cv2.absdiff(gray_1, gray_2)
cv2.imwrite("tmp_4.png", img_diff)

# 2値化
ret2, img_th = cv2.threshold(img_diff, 40, 255, cv2.THRESH_BINARY)
cv2.imwrite("tmp_5.png", img_th)

# 輪郭を検出
contours, hierarchy = cv2.findContours(
    img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# 閾値以上の差分を四角で囲う
for i, cnt in enumerate(contours):
    x, y, width, height = cv2.boundingRect(cnt)
    if width > 20 or height > 20:
        cv2.rectangle(img_1, (x, y), (x + width, y + height), (0, 0, 255), 1)

# 画像を生成
cv2.imwrite("./diff_image_res.png", img_1)
