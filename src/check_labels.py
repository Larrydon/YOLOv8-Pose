import os


def check_yolo_pose_labels(label_dir):
    print(f"正在檢查目錄: {label_dir}")
    files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    error_count = 0
    for filename in files:
        with open(os.path.join(label_dir, filename), "r") as f:
            lines = f.readlines()

        for line in lines:
            data = line.split()
            # YOLO Pose 格式: class x_center y_center width height k1_x k1_y k1_v k2_x k2_y k2_v ...
            # 關鍵點從 index 5 開始，每 3 個一組 (x, y, visibility)
            if len(data) < 5 + (4 * 3):
                continue

            # 提取 4 個關鍵點的 (x, y)
            kpts = []
            for i in range(4):
                idx = 5 + (i * 3)
                kpts.append((float(data[idx]), float(data[idx + 1])))

            p1, p2, p3, p4 = kpts  # 分別代表你標註的 1, 2, 3, 4 點

            errors = []
            # 邏輯檢查：
            # 1. 左上(p1) 的 x 應該小於 右上(p2) 的 x
            if p1[0] >= p2[0]:
                errors.append("點1(左上)不在點2(右上)的左邊")

            # 2. 左下(p4) 的 x 應該小於 右下(p3) 的 x
            if p4[0] >= p3[0]:
                errors.append("點4(左下)不在點3(右下)的左邊")

            # 3. 左上(p1) 的 y 應該小於 左下(p4) 的 y
            if p1[1] >= p4[1]:
                errors.append("點1(左上)不在點4(左下)的上方")

            # 4. 右上(p2) 的 y 應該小於 右下(p3) 的 y
            if p2[1] >= p3[1]:
                errors.append("點2(右上)不在點3(右下)的上方")

            if errors:
                error_count += 1
                print(f"❌ 檔案 {filename} 順序疑似有誤:")
                for e in errors:
                    print(f"   - {e}")

    if error_count == 0:
        print("✅ 恭喜！所有標註順序看起來都符合幾何邏輯。")
    else:
        print(f"\n總計找到 {error_count} 個可能有問題的標註檔。")


# --- 請修改下方的路徑 ---
train_label_path = "./dataset/train/labels"
val_label_path = "./dataset/val/labels"

check_yolo_pose_labels(train_label_path)
check_yolo_pose_labels(val_label_path)
