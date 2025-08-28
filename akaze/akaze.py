import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

DEBUG = "--debug" in sys.argv


def compositeImg(overlay, background, alpha=0.5):
    # overlay画像がRGBA（アルファチャンネル付き）でなければBGRに変換
    if len(overlay.shape) == 3 and overlay.shape[2] == 3:
        # 3チャンネルならアルファチャンネルを追加
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    # overlay画像をbackgroundと同じサイズにリサイズ（必要なら）
    overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))

    # アルファチャンネルを考慮して合成
    overlay_img = overlay[..., :3]  # 3次元配列の最後の次元の0,1,2番目
    overlay_mask = overlay[..., 3:] / 255.0 * alpha  # 最後の次元の3番目

    background = background.astype(float)
    overlay_img = overlay_img.astype(float)

    # 自然な合成のため、アルファ値の合計が1になるように合成
    blended = background * (1 - overlay_mask) + overlay_img * overlay_mask
    return blended.astype(np.uint8)


def alignImages(target_img, reference_img, good_match_ratio=0.15, min_matches=10):
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # [1] AKAZE特徴点検出器を作成し、特徴点と特徴量を検出
    akaze = cv2.AKAZE_create()
    keypoints_1, descriptors_1 = akaze.detectAndCompute(reference_gray, None)
    keypoints_2, descriptors_2 = akaze.detectAndCompute(target_gray, None)

    # [2] 特徴量同士を総当たりでマッチング
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(descriptors_1, descriptors_2)

    # マッチ結果を距離（類似度）でソート
    matches = sorted(matches, key=lambda x: x.distance)

    # DEBUG: マッチ結果を描画
    if DEBUG:
        img_matches = cv2.drawMatches(
            reference_img,
            keypoints_1,
            target_img,
            keypoints_2,
            matches[:20],
            None,
            flags=2,
        )
        cv2.imwrite("./matches.png", img_matches)

    # 良いマッチだけを抽出
    num_good_matches = int(len(matches) * good_match_ratio)
    good_matches = matches[:num_good_matches]

    # [3] 良いマッチが十分あれば、対応点リストを作成
    if len(good_matches) > min_matches:
        reference_pts_list = [keypoints_1[m.queryIdx].pt for m in good_matches]
        target_pts_list = [keypoints_2[m.trainIdx].pt for m in good_matches]

        # OpenCV用の形に変換 ex) [[[x1,y1]], [[x2,y2]], ...]
        reference_pts = np.float32(reference_pts_list).reshape(-1, 1, 2)
        target_pts = np.float32(target_pts_list).reshape(-1, 1, 2)

        # 対応点から射影変換行列（Homography）を計算
        homo_matrix, _ = cv2.findHomography(target_pts, reference_pts, cv2.RANSAC)

        # changed_imgをbase_imgに合わせて変形（ワープ）する
        height, width = reference_img.shape[:2]
        aligned_img = cv2.warpPerspective(target_img, homo_matrix, (width, height))
        return aligned_img, homo_matrix
    else:
        # マッチが少なすぎる場合は何も変化していないchanged_imgを返す
        return target_img, None


def plot_histograms(img, prefix="img"):
    # RGBヒストグラム
    colors = ("b", "g", "r")
    plt.figure(figsize=(10, 4))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title(f"{prefix} RGB Histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{prefix}_rgb_hist.png")
    plt.close()

    # HSVヒストグラム
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_labels = ("H", "S", "V")
    plt.figure(figsize=(10, 4))
    for i, label in enumerate(hsv_labels):
        hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
        plt.plot(hist, label=label)
        plt.xlim([0, 256])
    plt.title(f"{prefix} HSV Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_hsv_hist.png")
    plt.close()


def gamma_correction(img, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv2.LUT(img, table)


# 画像を読み込む
img_1 = cv2.imread("./img/set2_a.png")
img_2 = cv2.imread("./img/set2_b.png")
if img_1.shape != img_2.shape:
    img_1 = cv2.resize(img_1, (img_2.shape[1], img_2.shape[0]))
if DEBUG:
    print(img_1.shape, img_2.shape)

# 特徴量抽出、射影変換行列で片方の画像の大きさを揃える
aligned_img_1, homo_matrix = alignImages(img_1, img_2)

# 合成してちゃんとワープの形があっているか確認
if DEBUG:
    cv2.imwrite("composite.png", compositeImg(aligned_img_1, img_2))

# DEBUG: ヒストグラムをプロット
if DEBUG:
    plot_histograms(aligned_img_1, prefix="aligned_img_1")
    plot_histograms(img_2, prefix="img_2")

# グレースケール変換
gray_1 = cv2.cvtColor(aligned_img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# # ノイズ除去
# gray_1 = cv2.GaussianBlur(gray_1, (15, 15), 0)
# gray_2 = cv2.GaussianBlur(gray_2, (15, 15), 0)
# if DEBUG:
#     cv2.imwrite("blur_1.png", gray_1)
#     cv2.imwrite("blur_2.png", gray_2)

# diff = gray_1.mean() - gray_2.mean()
# gray_1 = cv2.add(gray_1, int(diff))
# gray_2 = cv2.add(gray_2, int(diff))


akaze = cv2.AKAZE_create()
keypoints_1, descriptors_1 = akaze.detectAndCompute(gray_1, None)
keypoints_2, descriptors_2 = akaze.detectAndCompute(gray_2, None)

# 特徴量をそのまま画像に貼り付けたい
img_1_with_keypoints = cv2.drawKeypoints(
    aligned_img_1, keypoints_1, None, color=(0, 255, 0)
)
img_2_with_keypoints = cv2.drawKeypoints(img_2, keypoints_2, None, color=(0, 255, 0))
if DEBUG:
    cv2.imwrite("img_1_with_keypoints.png", img_1_with_keypoints)
    cv2.imwrite("img_2_with_keypoints.png", img_2_with_keypoints)

# [2] 特徴量同士を総当たりでマッチング
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf_matcher.match(descriptors_1, descriptors_2)

if DEBUG:
    matched_queryIdx = set([m.queryIdx for m in matches])
    matched_trainIdx = set([m.trainIdx for m in matches])

    # img_1側でマッチしなかった特徴点
    unmatched_keypoints_1 = [
        kp for i, kp in enumerate(keypoints_1) if i not in matched_queryIdx
    ]
    # img_2側でマッチしなかった特徴点
    unmatched_keypoints_2 = [
        kp for i, kp in enumerate(keypoints_2) if i not in matched_trainIdx
    ]

    # 例：img_1にunmatched特徴点を赤で描画
    img_1_unmatched = cv2.drawKeypoints(
        aligned_img_1, unmatched_keypoints_1, None, color=(0, 0, 255)
    )
    cv2.imwrite("img_1_unmatched_keypoints.png", img_1_unmatched)

    # 例：img_2にunmatched特徴点を赤で描画
    img_2_unmatched = cv2.drawKeypoints(
        img_2, unmatched_keypoints_2, None, color=(0, 0, 255)
    )
    cv2.imwrite("img_2_unmatched_keypoints.png", img_2_unmatched)

# マッチ結果を距離（類似度）でソート
matches = sorted(matches, key=lambda x: x.distance)

# DEBUG: マッチ結果を描画
if DEBUG:
    img_matches = cv2.drawMatches(
        aligned_img_1,
        keypoints_1,
        img_2,
        keypoints_2,
        matches[:1000],
        None,
        flags=2,
    )
    cv2.imwrite("./matches_2.png", img_matches)
