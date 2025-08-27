import cv2
import numpy as np

img_1 = cv2.imread("./img/set2_a.png")
img_2 = cv2.imread("./img/set2_b.png")

if img_1.shape != img_2.shape:
    img_2 = cv2.resize(img_2, (img_1.shape[1], img_1.shape[0]))

# グレースケール変換
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
# ノイズ除去
gray_1 = cv2.GaussianBlur(gray_1, (5, 5), 0)
gray_2 = cv2.GaussianBlur(gray_2, (5, 5), 0)
# コントラスト限定適応ヒストグラム平坦化
# clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))
# gray_1 = clahe.apply(gray_1)
# gray_2 = clahe.apply(gray_2)

# エッジ検出
edges_1 = cv2.Canny(gray_1, 5, 100)
edges_2 = cv2.Canny(gray_2, 5, 100)
# edges_1 = cv2.Laplacian(gray_1, cv2.CV_8U, ksize=5)
# edges_2 = cv2.Laplacian(gray_2, cv2.CV_8U, ksize=5)

# harrisコーナー検出
corners_1 = cv2.cornerHarris(gray_1, 2, 3, 0.04)
corners_2 = cv2.cornerHarris(gray_2, 2, 3, 0.04)

corners_1 = cv2.dilate(corners_1, None)
corners_2 = cv2.dilate(corners_2, None)

corner_image_1 = edges_1.copy()
corner_image_1 = cv2.cvtColor(corner_image_1, cv2.COLOR_GRAY2BGR)
corner_image_1[corners_1 > 0.01 * corners_1.max()] = [0, 0, 255]  # 赤色でマーク
cv2.imwrite("corner_result_1.jpg", corner_image_1)
corner_image_2 = edges_2.copy()
corner_image_2 = cv2.cvtColor(corner_image_2, cv2.COLOR_GRAY2BGR)
corner_image_2[corners_2 > 0.01 * corners_2.max()] = [0, 0, 255]  # 赤色でマーク
cv2.imwrite("corner_result_2.jpg", corner_image_2)

# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(edges_1, None)
# kp2, des2 = orb.detectAndCompute(edges_2, None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)

# img_matches = cv2.drawMatches(edges_1, kp1, edges_2, kp2, matches[:20], None, flags=2)
# cv2.imwrite('./diff_image_res.png', img_matches)

# #二値化
# th_1 = cv2.adaptiveThreshold(edges_1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th_2 = cv2.adaptiveThreshold(edges_2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

# #画像を引き算
# img_diff = cv2.absdiff(th_1, th_2)

# #2値化
# ret2,img_th = cv2.threshold(img_diff,20,255,cv2.THRESH_BINARY)

# #輪郭を検出
# contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #閾値以上の差分を四角で囲う
# for i,cnt in enumerate(contours):
#     x, y, width, height = cv2.boundingRect(cnt)
#     if width > 20 or height > 20:
#         cv2.rectangle(img_1, (x, y), (x+width, y+height), (0, 0, 255), 1)

# #画像を生成
# cv2.imwrite("./diff_image_res.png", img_1)
