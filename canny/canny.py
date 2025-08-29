import cv2

# 画像をグレースケールで読み込む
gray_1 = cv2.imread("../img/set2_a.png", cv2.IMREAD_GRAYSCALE)
# Cannyエッジ検出
edges = cv2.Canny(gray_1, threshold1=1, threshold2=80)
cv2.imwrite("blur_1.png", edges)


# blur_1_dark = cv2.convertScaleAbs(gray_1, alpha=1.0, beta=-50)
# edges = cv2.Canny(blur_1_dark, threshold1=100, threshold2=200)
# cv2.imwrite("blur_1_dark.png", edges)


# blur_1_dark = cv2.convertScaleAbs(gray_1, alpha=1.0, beta=50)
# edges = cv2.Canny(blur_1_dark, threshold1=100, threshold2=200)
# cv2.imwrite("blur_1_dark_2.png", edges)


# blur_1_dark = cv2.convertScaleAbs(gray_1, alpha=-1.0, beta=-50)
# edges = cv2.Canny(blur_1_dark, threshold1=100, threshold2=200)
# cv2.imwrite("blur_1_dark_3.png", edges)

# blur_1_dark = cv2.convertScaleAbs(gray_1, alpha=3.0, beta=50)
# edges = cv2.Canny(blur_1_dark, threshold1=100, threshold2=200)
# cv2.imwrite("blur_1_dark_4.png", edges)

# 結果を保存


# 表示（任意）
# cv2.imshow("Edges", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
