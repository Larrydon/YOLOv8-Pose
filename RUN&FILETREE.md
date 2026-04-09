# 專案結構文件樹

- 專案結構文件樹
			(src）VSCode run Python
				|--[configs]	設定檔存放路徑
						|----coco8-pose.yaml	原始範例的設定檔
						|----license_plate_pose.yaml	修改成找車牌4頂點的設定檔
				|--[dataset]	資料集存放路徑
				|--[runs]
					|--[pose]	訓練、測試會自動產生的存放路徑
				|----check_labels.py	檢查標註檔4角方向是否有錯誤(1左上2右上3右下4左下) ./dataset/train/labels 和 ./dataset/val/labels
				|----convert_ppocr_to_yolo_pose.py	將PaddleOCR訓練用的總標註檔(Label.txt")轉成 YOLO格式(每個圖檔一個標註檔)
				|----predict.py	使用訓練好的 best.pt 預測圖檔
				|----refine_label.py	自動吸附4個頂點，並將結果可視化另存成圖片[images_refined]，藉以觀察標註4個頂點是否貼合
				|----YOLOv8-Pose.py	訓練的主程式
				|----yolov8s-pose.pt	版本模型，使用YOLOv8s的版本
 
- `.gitignore`
- `CHANGELOG.md`
- `README.md`
- `RUN&FILETREE.md`