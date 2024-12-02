import os
import json
import platform
import time
import pylab as plt
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFont, ImageDraw
from ultralytics import YOLO

def db(n:str):
    with open('./database.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    n = n.replace(' ', '').replace('\n', '')
    print(f"車牌號：{n}")
    
    stolen_cars = data['stolen_car']
    vip_cars = data['vip_car']

    if n in stolen_cars:
        print(f"警告！車牌號 {n} 是被盜車輛！")
    elif n in vip_cars:
        print(f"車牌號 {n} 是 VIP 車輛！")
    # print(stolen_cars)
    

def text(img, text, xy=(0, 0), color=(0, 0, 0), size=10):
    pil = Image.fromarray(img)
    s = platform.system()
    if s == "Linux":
        font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size)
    elif s == "Darwin":
        font = ImageFont.truetype('....', size)
    else:
        font = ImageFont.truetype('simsun.ttc', size)
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=color)
    return np.asarray(pil)

# 模型加載
model = YOLO('./car.pt')

# 打開攝影機
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # 讀取攝影機中的一幀
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # 轉換顏色格式
    img = frame[:, :, ::-1]  # BGR 轉換為 RGB
    
    # 進行車牌檢測
    results = model.predict(img, save=False)
    boxes = results[0].boxes.xyxy

    img_copy = np.copy(img)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # 確保座標是整數
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在圖片上繪製矩形

        # 車牌區域進行 OCR 識別
        tmp = cv2.cvtColor(img_copy[y1:y2, x1:x2].copy(), cv2.COLOR_RGB2GRAY)  # 複製區域
        license = pytesseract.image_to_string(tmp, lang='eng', config='--psm 11')

        ###
        db(license)
        ###
        
        # 在圖片上加上識別的車牌號
        frame = text(frame, license, (x1, y1 - 20), (0, 255, 0), 25)
        # frame = text(frame, license, (x1, y1 - 20), (0, 255, 0), 25) # 有邊框版本，但是字太大會打架
    
    # 顯示即時處理結果
    cv2.imshow("License Plate Detection", frame)

    # time.sleep(1)  # 每秒處理一次
    
    # 按下 'q' 退出循環
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉攝影機和窗口
cap.release()
cv2.destroyAllWindows()

