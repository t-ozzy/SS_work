# check_versions.py

# Pythonのバージョン情報を取得するためにsysをインポート
import sys

# 各ライブラリをインポート
try:
    import cv2
    import numpy
    import matplotlib
except ImportError as e:
    # ライブラリが一つでも見つからない場合にエラーメッセージを表示して終了
    print(f"エラー: 必要なライブラリが見つかりません。")
    print(f"詳細: {e}")
    print("pip install opencv-python numpy matplotlib を実行してインストールしてください。")
    sys.exit(1) # プログラムを異常終了させる

# バージョン情報を整形して出力
print("--- ライブラリのバージョン情報 ---")
# Pythonのバージョンはsysモジュールから取得
# sys.versionは詳細な情報を含むため、最初の1行だけ取得
print(f"Python:     {sys.version.splitlines()[0]}")
print(f"OpenCV:     {cv2.__version__}")
print(f"Numpy:      {numpy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print("---------------------------------")