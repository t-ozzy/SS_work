import cv2
import numpy as np


def alignImages(
    target_img, reference_img, max_pts=500, good_match_ratio=0.15, min_matches=10
):
    # 画像をグレースケールに変換（特徴点検出のため）
    gray_1 = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # [1] AKAZE特徴点検出器を作成し、特徴点と特徴量を検出
    akaze = cv2.AKAZE_create()
    keypoints_1, descriptors_1 = akaze.detectAndCompute(gray_1, None)
    keypoints_2, descriptors_2 = akaze.detectAndCompute(gray_2, None)

    # [2] 特徴量同士を総当たりでマッチング
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(descriptors_1, descriptors_2)

    # マッチ結果を距離（類似度）でソート
    matches = sorted(matches, key=lambda x: x.distance)

    # # DEBUG: マッチ結果を描画
    # img_matches = cv2.drawMatches(gray_1, keypoints_1, gray_2, keypoints_2, matches[:10], None, flags=2)
    # cv2.imwrite("./akaze_matches.png", img_matches)

    # 良いマッチだけを抽出
    num_good_matches = int(len(matches) * good_match_ratio)
    good_matches = matches[:num_good_matches]

    # [3] 良いマッチが十分あれば、対応点リストを作成
    if len(good_matches) > min_matches:
        reference_pts_list = []
        target_pts_list = []
        for match in good_matches:
            reference_pts_list.append(
                keypoints_1[match.queryIdx].pt
            )  # reference_img側の点
            target_pts_list.append(keypoints_2[match.trainIdx].pt)  # target_img側の点

        # OpenCV用の形に変換 ex) [[[x1,y1]], [[x2,y2]], ...]
        reference_pts = np.float32(reference_pts_list).reshape(-1, 1, 2)
        target_pts = np.float32(target_pts_list).reshape(-1, 1, 2)

        # 対応点から射影変換行列（Homography）を計算
        homography_matrix, mask = cv2.findHomography(
            target_pts, reference_pts, cv2.RANSAC
        )

        # changed_imgをbase_imgに合わせて変形（ワープ）する
        height, width, channels = reference_img.shape
        aligned_img = cv2.warpPerspective(
            target_img, homography_matrix, (width, height)
        )
        return aligned_img, homography_matrix
    else:
        # マッチが少なすぎる場合は何も変化していないchanged_imgを返す
        return target_img, None


# 画像を読み込む
source_img = cv2.imread("../img/set2_a.png")
target_img = cv2.imread("../img/set2_b.png")

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
