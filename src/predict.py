from ultralytics import YOLO
import numpy as np
import cv2




# 載入練好的模型
model = YOLO('runs/pose/train3/weights/best.pt')

# 預測
results = model.predict(source='test_car.jpg', save=True, conf=0.5)


# 顯示結果
#results[0].show()

# 3. 查看關鍵點座標
for r in results:
    # 獲取關鍵點座標 (x, y)
    # kpts 格式通常為 [1, 4, 2] -> 1 個車牌, 4 個點, (x, y)
    kpts = r.keypoints.xy.cpu().numpy()
    print("車牌四個角點座標：\n", kpts)


# 假設 r 是 results 中的一個物件
for r in results:
    img = r.orig_img.copy() # 複製原圖
    
    # 畫框 (Bbox) - 用藍色
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # 藍色框

    # 畫關鍵點 (Keypoints)
    if r.keypoints is not None:
        # 取得座標
        kpts = r.keypoints.xy[0].cpu().numpy()
        
        # 定義顏色清單 (BGR 格式)
        # 第一個點：紅色, 第二個：綠色, 第三個：黃色, 第四個：紫色
        colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        for i, (kx, ky) in enumerate(kpts):
            if kx > 0 and ky > 0: # 確保點位有效
                # 畫圓點
                cv2.circle(img, (int(kx), int(ky)), 5, colors[i % len(colors)], -1)
                # 標註序號，讓你更清楚順序
                cv2.putText(img, str(i+1), (int(kx), int(ky)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)

    cv2.imwrite('custom_predict.jpg', img)


for r in results:
    # 檢查是否有抓到關鍵點
    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        # 取得第一組關鍵點 (假設只有一個車牌)
        src_pts = r.keypoints.xy[0].cpu().numpy().astype("float32")
        
        # 必須正好是 4 個點才能做透視變換
        if len(src_pts) == 4:
            # 定義拉直後的尺寸
            width, height = 300, 100
            
            # 定義目標座標 (左上, 右上, 右下, 左下)
            dst_pts = np.array([
                [0, 0], 
                [width, 0], 
                [width, height], 
                [0, height]
            ], dtype="float32")
            
            # 計算轉換矩陣並校正圖片
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(r.orig_img, M, (width, height))
            
            # 儲存結果
            cv2.imwrite('check_warp.jpg', warped)
            print("--- 恭喜！校正後的圖片已儲存為 check_warp.jpg ---")
        else:
            print(f"偵測到的點數不等於 4 (抓到 {len(src_pts)} 個點)")
    else:
        print("未偵測到任何車牌或關鍵點")