import cv2
import numpy as np

img = cv2.imread("./img/set2_a.png")
if img is None:
    raise FileNotFoundError("画像が見つかりません")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ノイズ除去
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# エッジ検出
edges = cv2.Canny(blurred, 60, 100, apertureSize=3)

# ハフ変換で直線検出
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=30, maxLineGap=20)
print(lines)

# 検出した直線を描画
line_img = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("hough_lines_result.png", line_img)