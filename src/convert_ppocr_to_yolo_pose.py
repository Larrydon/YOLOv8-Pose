import json
import os
import cv2


def convert():
    # ================= 配置區域 =================
    image_dir = (
        r"F:\UserData\Larry\Documents\VSCode Project\Python\testIMG"  # 放照片的資料夾
    )
    label_file = r"F:\UserData\Larry\Documents\VSCode Project\Python\testIMG\Label.txt"  # PPOCRLabel 輸出的總標註檔
    output_dir = r"F:\UserData\Larry\Documents\VSCode Project\Python\YOLOv8-Pose\dataset"  # 預計輸出的 YOLO 標籤資料夾
    # ===========================================

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 讀取 PPOCRLabel 的 Label.txt
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        # PPOCR 格式通常是: 檔名\t[{"transcription": "...", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        img_name = parts[0]
        # 取得純檔名 (不含路徑與副檔名)
        file_id = os.path.splitext(os.path.basename(img_name))[0]

        # 讀取圖片取得寬高
        img_path = os.path.join(image_dir, os.path.basename(img_name))
        img = cv2.imread(img_path)
        if img is None:
            print(f"找不到圖片: {img_path}")
            continue
        h, w, _ = img.shape

        annotations = json.loads(parts[1])
        yolo_results = []

        for anno in annotations:
            pts = anno["points"]  # 這是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

            # 1. 計算 Bounding Box (x_center, y_center, width, height) 並正規化
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            bx = (xmin + xmax) / 2 / w
            by = (ymin + ymax) / 2 / h

            # 2. 格式化關鍵點 (px, py, pv) pv=2 代表可見
            # YOLO-pose 格式: class_id bx by bw bh px1 py1 pv1 px2 py2 pv2 ...
            kpts_str = ""
            for p in pts:
                kpts_str += f"{p[0]/w} {p[1]/h} 2 "

            # 假設類別只有一個 (車牌)，ID 為 0
            yolo_results.append(f"0 {bx} {by} {bw} {bh} {kpts_str.strip()}")

        # 寫出到對應的 TXT 檔
        with open(os.path.join(output_dir, f"{file_id}.txt"), "w") as f_out:
            f_out.write("\n".join(yolo_results))

    print(f"轉換完成！標籤已存至: {output_dir}")


if __name__ == "__main__":
    convert()
