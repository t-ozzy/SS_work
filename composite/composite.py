import cv2
import numpy as np


def compositeImg(overlay, background):
    # overlay画像がRGBA（アルファチャンネル付き）でなければBGRに変換
    if overlay.shape[2] == 3:
        # 3チャンネルならアルファチャンネルを追加
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    # 透過率を設定（0.5 = 50%）
    alpha = 0.5

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


# 画像を読み込む
background = cv2.imread("../img/set2_b.png")
overlay = cv2.imread(
    "../projective_transformation/aligned_image.png", cv2.IMREAD_UNCHANGED
)

# 合成
blended = compositeImg(overlay, background)

# 保存
cv2.imwrite("result.jpg", blended)
