import cv2
import numpy as np


def alignImages(img1, img2, max_pts=500, good_match_rate=0.15, min_match=10):
    # 画像をグレースケールに変換（特徴点検出のため）
    gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # [1] AKAZE特徴点検出器を作成し、特徴点と特徴量を検出
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray_1, None)
    kp2, des2 = akaze.detectAndCompute(gray_2, None)

    # [2] 特徴量同士を総当たりでマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # マッチ結果を距離（類似度）でソート
    matches = sorted(matches, key=lambda x: x.distance)

    # # DEBUG: マッチ結果を描画
    # img_matches = cv2.drawMatches(gray_1, kp1, gray_2, kp2, matches[:10], None, flags=2)
    # cv2.imwrite("./akaze_matches.png", img_matches)

    # 良いマッチだけを抽出
    good = matches[: int(len(matches) * good_match_rate)]

    # [3] 良いマッチが十分あれば、対応点リストを作成
    if len(good) > min_match:
        src_pts_list = []
        dst_pts_list = []
        for m in good:
            src_pts_list.append(kp1[m.queryIdx].pt)  # img1側の点
            dst_pts_list.append(kp2[m.trainIdx].pt)  # img2側の点

        # OpenCV用の形に変換 ex) [[[x1,y1]], [[x2,y2]], ...]
        src_pts = np.float32(src_pts_list).reshape(-1, 1, 2)
        dst_pts = np.float32(dst_pts_list).reshape(-1, 1, 2)

        # 対応点から射影変換行列（Homography）を計算 #Q Homographyって何？
        h, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

        # img2をimg1に合わせて変形（ワープ）する
        height, width, channels = img1.shape
        dst_img = cv2.warpPerspective(img2, h, (width, height))
        return dst_img, h
    else:
        # マッチが少なすぎる場合は変形せず元画像を返す
        return img1, np.zeros((3, 3))


# 画像を読み込む
img_1 = cv2.imread("./img/set2_a.png")
img_2 = cv2.imread("./img/set2_b.png")
if img_1.shape != img_2.shape:
    img_1 = cv2.resize(img_1, (img_2.shape[1], img_2.shape[0]))


# 特徴量抽出、射影変換行列で片方の画像の大きさを揃える
aligned_img, homography = alignImages(img_2, img_1)

# 画像に変化を加える際の前準備をする
# グレースケール変換
gray_1 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
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
## ガンマ補正で明るさを調整
# gamma = 1.8  # この値で明るさが変わる
# gamma_cvt = np.zeros((256, 1), dtype="uint8")
# for i in range(256):
#     gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
# img_1 = cv2.LUT(img_1, gamma_cvt)
# img_2 = cv2.LUT(img_2, gamma_cvt)


# 画像を比較する
# 画像を引き算
img_diff = cv2.absdiff(gray_1, gray_2)
cv2.imwrite("tmp_4.png", img_diff)
# 2値化
ret2, img_th = cv2.threshold(img_diff, 40, 255, cv2.THRESH_BINARY)
cv2.imwrite("tmp_5.png", img_th)

# 結果を出力する
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
cv2.imwrite("./diff_image_res_2.png", img_1)
