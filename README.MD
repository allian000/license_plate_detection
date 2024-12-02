# Yolov8 車牌辨識
這是一個簡單的車牌辨識模型，可以偵測車牌並將車牌裁切，透過 `pytesseract` 來進行 OCR。

## 安裝
下鮺這個專案
```shell
git clone https://github.com/allian000/license_plate_detection.git
```

安裝必要套件
```shell
pip install requirements.txt
```

## 訓練模型

> [!NOTE]
> 如果你想要訓練自己的資料集，你需要自己替換 `data.yaml` 中的設定。

```shell
python train.py
```

## 執行程式
```shell
python main.py
```