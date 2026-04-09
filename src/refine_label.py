import cv2
import numpy as np
import os


#BASE_DIR = "./dataset/train"
BASE_DIR = "./dataset/val"


def refine_label(img_path, txt_path, save_path, debug_img_path):
    # 1. 讀取圖片與標註
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]

    # 複製一份圖片用來畫畫 (Debug 用)
    debug_img = img.copy()

    with open(txt_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        data = line.split()
        if len(data) < 17:
            new_lines.append(line)
            continue

        # 提取原始 4 個點 (歸一化座標轉像素座標)
        raw_pts = []
        for i in range(4):
            raw_pts.append([float(data[5 + i * 3]) * w, float(data[6 + i * 3]) * h])

        refined_pts_px = []  # 儲存像素座標供繪圖用
        refined_pts_norm = []  # 儲存歸一化座標供存檔用

        # 1. 先計算這四個點的中心 (中心點是絕對安全的搜尋方向)
        cx = sum(p[0] for p in raw_pts) / 4
        cy = sum(p[1] for p in raw_pts) / 4

        # 搜尋視窗大小 (像素，可以設大一點，例如 15-20，讓它有空間移動)
        patch_size = 5

        for i, (px, py) in enumerate(raw_pts):
            # 2. 讓搜尋視窗的中心向車牌中心「偏移」5 像素
            # 這樣視窗會包含更多車牌內容，減少外部背景干擾
            offset_x = 5 if px < cx else -5
            offset_y = 5 if py < cy else -5

            search_x = px + offset_x
            search_y = py + offset_y

            # 2. 切出局部小塊 (Patch)
            x1, y1 = max(0, int(search_x - patch_size)), max(
                0, int(search_y - patch_size)
            )
            x2, y2 = min(w, int(search_x + patch_size)), min(
                h, int(search_y + patch_size)
            )
            patch = img[y1:y2, x1:x2]

            # 3. 影像處理
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
            # # 使用自適應二值化，應對不同光照，通常是找白色最大區域
            # thresh = cv2.adaptiveThreshold(
            #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1
            # )
            #
            # # 尋找輪廓
            # contours, _ = cv2.findContours(
            #     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            # )
            #
            #
            # 3. 改用邊緣偵測，不分紅底白底
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            # 先模糊化減少雜訊
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            # Canny 偵測邊緣
            edged = cv2.Canny(blurred, 10, 150)
            
            # 膨脹一下線條，讓斷掉的邊框連起來
            kernel = np.ones((3,3), np.uint8)
            edged = cv2.dilate(edged, kernel, iterations=1)

            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # final_px, final_py = px, py  # 預設原點
            # if contours:
            #     # 找面積最大的 (通常是車牌白框的一角)
            #     c = max(contours, key=cv2.contourArea)
            #     # 找出該輪廓中距離原始點最近的極致點
            #     # 這裡簡化為：根據點的順序找最角落的座標
            #     pts_in_patch = c.reshape(-1, 2)

            #     # 3. 改變邏輯：向中心點「吸附」
            #     # 左上角(i=0)要找局部區域中最靠「右下」的點，其餘類推
            #     if i == 0:  # 左上：找 patch 裡最靠右下的點
            #         best = pts_in_patch[
            #             np.argmax(pts_in_patch[:, 0] + pts_in_patch[:, 1])
            #         ]
            #     elif i == 1:  # 右上：找 patch 裡最靠左下的點
            #         best = pts_in_patch[
            #             np.argmin(pts_in_patch[:, 0] - pts_in_patch[:, 1])
            #         ]
            #     elif i == 2:  # 右下：找 patch 裡最靠左上的點
            #         best = pts_in_patch[
            #             np.argmin(pts_in_patch[:, 0] + pts_in_patch[:, 1])
            #         ]
            #     elif i == 3:  # 左下：找 patch 裡最靠右上的點
            #         best = pts_in_patch[
            #             np.argmax(pts_in_patch[:, 0] - pts_in_patch[:, 1])
            #         ]

            #     final_px, final_py = best[0] + x1, best[1] + y1
            
            
            final_px, final_py = px, py 
            if contours:
                # 找「最靠近原始點」的輪廓，而不是「最大」的輪廓
                # 這樣可以避免被遠處的雜訊吸走
                c = max(contours, key=cv2.contourArea) 
                pts_in_patch = c.reshape(-1, 2)
               
                # 這裡維持你的 patch_size = 5 的精細度
                if i == 0:  # 左上：找最左上的點
                    best = pts_in_patch[np.argmin(pts_in_patch[:, 0] + pts_in_patch[:, 1])]
                elif i == 1:    # 右上：找最右上的點
                    best = pts_in_patch[np.argmax(pts_in_patch[:, 0] - pts_in_patch[:, 1])]
                elif i == 2:    # 右下：找最右下的點
                    best = pts_in_patch[np.argmax(pts_in_patch[:, 0] + pts_in_patch[:, 1])]
                elif i == 3:    # 左下：找最左下的點
                    best = pts_in_patch[np.argmin(pts_in_patch[:, 0] - pts_in_patch[:, 1])]
               
                final_px, final_py = best[0] + x1, best[1] + y1

            # 轉換回原圖座標
            refined_pts_px.append((int(final_px), int(final_py)))
            refined_pts_norm.append([final_px / w, final_py / h])

        # --- 繪製視覺化成果 ---
        # 畫出四個點的連線 (封閉多邊形)
        pts_arr = np.array(refined_pts_px, np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            debug_img, [pts_arr], isClosed=True, color=(255, 255, 0), thickness=2
        )

        # 畫出點位並標註序號
        for idx, pt in enumerate(refined_pts_px):
            cv2.circle(debug_img, pt, 5, (0, 0, 255), -1)
            cv2.putText(
                debug_img,
                str(idx + 1),
                (pt[0], pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # 5. 組裝回 YOLO 格式
        new_data = data[:5]
        for i in range(4):
            new_data.extend(
                [f"{refined_pts_norm[i][0]:.6f}", f"{refined_pts_norm[i][1]:.6f}", "2"]
            )
        new_lines.append(" ".join(new_data) + "\n")

    # 存下標註後的圖片
    cv2.imwrite(debug_img_path, debug_img)

    with open(save_path, "w") as f:
        f.writelines(new_lines)


# --- 執行批次處理 ---
img_dir = os.path.join(BASE_DIR, "images")
txt_dir = os.path.join(BASE_DIR, "labels")
out_txt_dir = os.path.join(BASE_DIR, "labels_refined")
out_img_dir = os.path.join(BASE_DIR, "images_refined")  # 新增的圖片儲存路徑

if not os.path.exists(out_txt_dir):
    os.makedirs(out_txt_dir)
if not os.path.exists(out_img_dir):
    os.makedirs(out_img_dir)

for f in os.listdir(txt_dir):
    if f.endswith(".txt"):
        img_f = f.replace(".txt", ".jpg")
        if os.path.exists(os.path.join(img_dir, img_f)):
            refine_label(
                os.path.join(img_dir, img_f),
                os.path.join(txt_dir, f),
                os.path.join(out_txt_dir, f),
                os.path.join(out_img_dir, img_f),  # 存到 re_images
            )

print(f"🎉 全部校正完成！")
print(f"👉 修正後的標註檔在: {out_txt_dir}")
print(f"👉 視覺化成果圖在: {out_img_dir}")
