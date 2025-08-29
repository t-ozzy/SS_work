import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

DEBUG = "--debug" in sys.argv


def compositeGrayImg(overlay, background, alpha=0.5):
    # overlay画像をbackgroundと同じサイズにリサイズ
    overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))

    # 画像をfloat型に変換（計算のため）
    overlay = overlay.astype(float)
    background = background.astype(float)

    # 合成処理（アルファブレンディング）
    blended = background * (1 - alpha) + overlay * alpha

    # 結果をuint8型に戻す
    return blended.astype(np.uint8)


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
img_1 = cv2.imread("../img/set2_a.png")
img_2 = cv2.imread("../img/set2_b.png")
if img_1.shape != img_2.shape:
    img_1 = cv2.resize(img_1, (img_2.shape[1], img_2.shape[0]))

# 特徴量抽出、射影変換行列で片方の画像の大きさを揃える
aligned_img_1, homo_matrix = alignImages(img_1, img_2)

# 合成してちゃんとワープの形があっているか確認
if DEBUG:
    cv2.imwrite("composite.png", compositeImg(aligned_img_1, img_2))

# DEBUG: ヒストグラムをプロット
if DEBUG:
    plot_histograms(aligned_img_1, prefix="aligned_img_1")
    plot_histograms(img_2, prefix="img_2")

# hsv1[..., 1] = np.clip(hsv1[..., 1] * 1.5, 0, 255)  # 彩度を1.5倍に

# hsv2[..., 1] = np.clip(hsv2[..., 1] * 1.5, 0, 255)  # 彩度を1.5倍に

# s_img_1 = cv2.cvtColor(aligned_img_1, cv2.COLOR_BGR2HSV)
# s_img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2HSV)
# s_img_1[:, :, 1] = 250
# s_img_2[:, :, 1] = 250
# c_aligned_img_1 = cv2.cvtColor(s_img_1, cv2.COLOR_HSV2BGR)
# c_img_2 = cv2.cvtColor(s_img_2, cv2.COLOR_HSV2BGR)

# グレースケール変換
gray_1 = cv2.cvtColor(aligned_img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# # ノイズ除去
# gray_1 = cv2.GaussianBlur(gray_1, (15, 15), 0)
# gray_2 = cv2.GaussianBlur(gray_2, (15, 15), 0)
# if DEBUG:
#     cv2.imwrite("blur_1.png", gray_1)
#     cv2.imwrite("blur_2.png", gray_2)

edges_1 = cv2.Canny(gray_1, threshold1=40, threshold2=150)
edges_2 = cv2.Canny(gray_2, threshold1=40, threshold2=150)
edges_result = cv2.absdiff(edges_2, edges_1)
cv2.imwrite("edges_diff.png", edges_result)

# tmp = gray_1.copy()
# mask = np.full(img_2.shape[:2], 255, dtype=np.uint8)
# warped_mask = cv2.warpPerspective(mask, homo_matrix, (img_2.shape[1], img_2.shape[0]))
# warped_mask = cv2.erode(warped_mask, (5, 5), iterations=10)
# tmp[warped_mask == 0] = 0

# コントラスト限定適応ヒストグラム平坦化
# gray_1 = cv2.equalizeHist(tmp)
# gray_2 = cv2.equalizeHist(gray_2)
clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
gray_1 = clahe.apply(gray_1)
gray_2 = clahe.apply(gray_2)
if DEBUG:
    cv2.imwrite("clahe_1.png", gray_1)
    cv2.imwrite("clahe_2.png", gray_2)

# 画像を引き算
img_diff = cv2.absdiff(gray_1, gray_2)
if DEBUG:
    cv2.imwrite("diff.png", img_diff)

# 2値化
# img_th = cv2.adaptiveThreshold(
#     img_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )
_, img_th = cv2.threshold(img_diff, 40, 255, cv2.THRESH_BINARY)  # 閾値40は手作業
# _, img_th = cv2.threshold(img_diff, 0, 255, cv2.THRESH_OTSU)
if DEBUG:
    cv2.imwrite("th.png", img_th)

img_th = cv2.medianBlur(img_th, 5)

# 有効領域マスクを作成
if homo_matrix is not None:
    mask = np.full(img_2.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        mask, homo_matrix, (img_2.shape[1], img_2.shape[0])
    )
    # マスクを収縮させて削る面積を増やす
    warped_mask = cv2.erode(warped_mask, (5, 5), iterations=10)
    # 二値画像の余白を黒で塗りつぶす
    img_th[warped_mask == 0] = 0

if DEBUG:
    cv2.imwrite("after.png", img_th)

# 輪郭を検出
contours, hierarchy = cv2.findContours(
    img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# 閾値以上の差分を四角で囲う
# for i, cnt in enumerate(contours):
#     x, y, width, height = cv2.boundingRect(cnt)
#     if width > 20 or height > 20:
#         cv2.rectangle(img_1, (x, y), (x + width, y + height), (0, 0, 255), 2)
cv2.drawContours(aligned_img_1, contours, -1, (0, 0, 255), 2)

# 画像を生成
cv2.imwrite("./result.png", aligned_img_1)
