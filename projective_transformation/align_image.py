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
        dst_img = cv2.warpPerspective(img2, homography, (width, height))
        return dst_img, homography
    else:
        # マッチが少なすぎる場合は何も変化していないimg2を返す
        return img2, None


# 画像を読み込む
source_img = cv2.imread("./img/set2_a.png")
target_img = cv2.imread("./img/set2_b.png")

print(target_img.shape)  # 画像サイズの確認

# gamma = 1.8  # この値で明るさが変わる
# gamma_cvt = np.zeros((256, 1), dtype="uint8")
# for i in range(256):
#     gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)

# # ガンマ補正を適用
# target_img = cv2.LUT(target_img, gamma_cvt)
# source_img = cv2.LUT(source_img, gamma_cvt)

# 画像を位置合わせ
aligned_img, homography = alignImages(target_img, source_img)

print(aligned_img.shape)  # 結果画像のサイズ確認

# 結果を保存
cv2.imwrite("aligned_image.png", aligned_img)
